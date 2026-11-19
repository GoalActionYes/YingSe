"""Qualify a fixed Film B edit on existing real background-swap photographs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, quantize
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256, grade
from probe_frozen_render_blend_bound import FULL, FULL_SHA
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_paired_cubic_cross_content import apply, paired_fit
from probe_target_mirror_chroma import checked_rgb, panel, save
from spatial_render import SpatialEngine, finish_spatial
from verify_development_inputs import manifest, verify


FILM_SHA = "ef7a8ed92baf4109236131bbb51b3a55a9bbeaff980577f2df845d418096a30a"
ROLES = ("C1", "F1")


def transform(rgb, processors):
    display, inverse, film = processors
    log = converted(rgb.astype(np.float32) / 255, inverse)
    off = quantize(np.clip(converted(log, display), 0, 1))
    on = converted(converted(log, film), display)
    return quantize(np.clip(on, 0, 1)), {
        "roundtrip_max_rgb8": int(np.abs(off.astype(np.int16) - rgb).max()),
        "raw_min": float(on.min()), "raw_max": float(on.max()),
        "out_of_gamut_channel_fraction": float(((on < 0) | (on > 1)).mean()),
    }


def prepare(out):
    cube = BASE / "arri-second-film-assets-v1/second-film.cube"
    if digest(cube) != FILM_SHA:
        raise ValueError("Film B table changed")
    processors = transforms(cube)
    entries = {item["pair"]: item for item in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    cases = {}
    for name, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        folder = BASE / "bg-real-editor-look" / name
        files = row["artifacts"]
        target = checked_rgb(folder / "target.png", files["target.png"])
        person = mask_for(entries[name], name.split("-")[0])
        raw, diagnostic = transform(target, processors)
        known = finish_spatial(target, raw, STRENGTH)
        expected = {"off": target, "on": known}
        references, reference_diagnostics = {}, {}
        for role in ROLES:
            filename = f"{role}-off-reference.png"
            original = checked_rgb(folder / filename, files[filename])
            edited, detail = transform(original, processors)
            references[role] = {"off": original, "on": edited}
            reference_diagnostics[role] = {**detail, "edited_sha256": save(out / f"{name}-{role}-reference.png", edited)}
        oracle_geometry = evaluate(target, expected, dict.fromkeys(ROLES, expected), person)
        cases[name] = (target, expected, person, references, {
            "eligible": oracle_geometry["passed"], "oracle_geometry": oracle_geometry,
            "target_diagnostic": diagnostic, "reference_diagnostics": reference_diagnostics,
            "expected_sha256": save(out / f"{name}-expected.png", known),
        })
    return cases


def score(cases, out):
    parent = verified_json(BASE / FULL / "results.json", FULL_SHA)["suites"]["background"]["cases"]
    original = manifest(BASE, "bg-real-editor-look")["cases"]
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Frozen model identity changed")
    results = {}
    for mode in ("13", "expanded", "paired"):
        if mode == "expanded":
            initialize_replication_model(engine.model, engine.model.state_dict())
            engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
            engine.model.eval().requires_grad_(False)
        rows = {}
        for name, (target, expected, person, references, _) in cases.items():
            rendered, audit, board = {}, {}, {}
            for role, variants in references.items():
                if mode == "paired":
                    table, detail = paired_fit(variants["off"], variants["on"])
                    rendered[role] = {"off": target, "on": apply(target, table)}
                    audit[role] = detail
                else:
                    if mode == "13":
                        rendered[role] = {state: finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
                                          for state, ref in variants.items()}
                        filename = f"{role}-off-13.png"
                        archive = checked_rgb(BASE / "bg-real-editor-look" / name / filename,
                                              original[name]["artifacts"][filename])
                    else:
                        rendered[role] = {state: grade(engine, target, ref, target)[0] for state, ref in variants.items()}
                        archive = checked_rgb(BASE / FULL / "background" / name / f"{role}-off.png",
                                              parent[name]["outputs"][f"{role}-off"]["sha256"])
                    np.testing.assert_array_equal(rendered[role]["off"], archive)
                for state, output in rendered[role].items():
                    board[f"{role}-{state}"] = {"target": target, "reference": variants[state],
                                                mode: output, "expected": expected[state]}
            rows[name] = {
                "scores": evaluate(target, expected, rendered, person), "reference_fit": audit,
                "outputs": {f"{role}-{state}": save(out / f"{mode}-{name}-{role}-{state}.png", image)
                            for role, states in rendered.items() for state, image in states.items()},
                "comparison_sha256": panel(out / f"{mode}-{name}-comparison.png", board),
            }
        count = sum(row["scores"]["passed"] for row in rows.values())
        worst = max(row["scores"]["max_protected_swap"] for row in rows.values())
        results[mode] = {"cases": rows, "passing_cases": count, "max_protected_swap": worst,
                         "passed": count >= 3 and worst <= 3}
    return results


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    out.mkdir()
    cases = prepare(out)
    admission = {name: row[-1] for name, row in cases.items()}
    count = sum(row["eligible"] for row in admission.values())
    results = score(cases, out) if count >= 3 else None
    json_to(out / "results.json", {"film_sha256": FILM_SHA, "strength": STRENGTH, "qualification": admission,
                                   "qualified_cases": count, "qualified": count >= 3, "comparison": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
