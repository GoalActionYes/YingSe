"""Exercise the current 13 app store with matched and cross-content photos."""

import argparse
from pathlib import Path

import numpy as np
import torch

from probe_arri_film_b_dose import endpoints
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH
from probe_frozen_same_content_control import build_cases, measure
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, panel, save
from reference_color_app.aligned_reference import fit_aligned
from spatial_render import SpatialEngine
from spatial_store import SpatialStore
from verify_development_inputs import manifest, verify

RAW = "frozen-same-content-control-v2"
RAW_SHA = "6c316efe4692b7fd67f11335ffb80fdf5be55d1157ff25dc70a0960a36224c92"


def cross_certification(cases):
    bg = manifest(BASE, "bg-real-editor-look")["cases"]
    prep = manifest(BASE, "arri-two-looks-v2")
    pairs = manifest(BASE, "arri-film-b-crossed-v1")["cases"].values()
    results = {}
    for suite, name, target, *_ in cases:
        if suite == "background":
            files = bg[name]["artifacts"]
            root = BASE / "bg-real-editor-look" / name
            refs = {f"{variant}-{role}": checked_rgb(root / f"{variant}-{role}-reference.png",
                    files[f"{variant}-{role}-reference.png"])
                    for variant in ("C1", "F1") for role in ("off", "on")}
        else:
            refs = {f"{row['reference']}-{role}": rgb for row in pairs if row["target"] == name
                    for role, rgb in zip(("off", "on"), endpoints(BASE / "arri-two-looks-v2", prep,
                                                                   row["reference"]), strict=True)}
        results[name] = {label: fit_aligned(target, ref) for label, ref in refs.items()}
    return results


def run(out):
    verify(BASE, COUNTS)
    source = verified_json(BASE / RAW / "results.json", RAW_SHA)
    cases = list(build_cases())
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    out.mkdir()
    results = {"background": {}, "film_b": {}}
    store = SpatialStore(out / "store", engine)
    for suite, name, target, edited, expected, protected in cases:
        prefix = f"{suite}-{name}"
        target_file, styled_file = (out / f"{prefix}-{role}-input.png" for role in ("off", "on"))
        save(target_file, target)
        save(styled_file, edited)
        target_entry = store.add_media(target_file)
        edited_entry = store.add_media(styled_file)
        decoded, _ = store.decode(target_entry)
        np.testing.assert_array_equal(decoded, target)
        renders, records, panels = {}, {}, {}
        for state, ref in (("off", target_entry), ("on", edited_entry)):
            mapping = store.analyze(target_entry["id"], ref["id"])
            diagnostic = mapping["diagnostics"]
            identifier = mapping["id"]
            preview = store.render(identifier, decoded, STRENGTH, preview=True)
            normal = store.render(identifier, decoded, STRENGTH)
            np.testing.assert_array_equal(preview, normal)
            raw_case = source["suites"][suite][name]["13"]
            raw = checked_rgb(BASE / RAW / f"{prefix}-13-{state}.png", raw_case["images"][state])
            certification = diagnostic["certified_aligned"]
            if state == "off":
                np.testing.assert_array_equal(normal, target)
                if not diagnostic["exact_identity"]:
                    raise ValueError("Identity was not recorded")
            elif not certification["accepted"]:
                np.testing.assert_array_equal(normal, raw)
            renders[state] = normal
            records[state] = {
                "exact_identity": diagnostic["exact_identity"], "certification": certification,
                "preview_matches_render": True, "output_sha256": save(out / f"{prefix}-{state}.png", normal),
                "vs_raw_mae_rgb8": float(np.abs(normal.astype(np.int16) - raw).mean()),
            }
            panels[state] = {"target": target, "reference": target if state == "off" else edited,
                             "native": raw, "normal": normal, "expected": target if state == "off" else expected}
        results[suite][name] = {
            "states": records, "metrics": measure(suite, target, expected, renders, protected),
            "panel_sha256": panel(out / f"{prefix}-comparison.png", panels),
        }
    cross = cross_certification(cases)
    json_to(out / "results.json", {"source_sha256": RAW_SHA, "checkpoint_sha256": CHECKPOINT, "strength": STRENGTH,
                                   "same_content": results, "cross_content_certification": cross})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
