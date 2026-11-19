"""Measure native style response when the reference shares the target pixels."""

import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_arri_film_b_dose import endpoints
from probe_arri_shared_look import metrics
from probe_bg_real_editor_look import PARENT, PARENT_SHA, authored, mask_for, split_signal
from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256
from probe_frozen_render_blend_bound import FULL, FULL_SHA, neutral
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import manifest, verify


def build_cases():
    parent = verified_json(PARENT / "results.json", PARENT_SHA)
    entries = {row["pair"]: row for row in parent["cases"]}
    for name, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        files = row["artifacts"]
        root = BASE / "bg-real-editor-look" / name
        target, expected = (checked_rgb(root / x, files[x]) for x in ("target.png", "expected-on.png"))
        edited = authored(target)
        np.testing.assert_array_equal(expected, finish_spatial(target, edited, STRENGTH))
        person = mask_for(entries[name], name.split("-")[0])
        yield "background", name, target, edited, expected, neutral(target) & person
    prep = manifest(BASE, "arri-two-looks-v2")
    for name in prep["sources"]:
        target, edited = endpoints(BASE / "arri-two-looks-v2", prep, name)
        yield "film_b", name, target, edited, finish_spatial(target, edited, STRENGTH), neutral(target) & neutral(edited)


def background_metrics(target, expected, off, on):
    wanted = split_signal(expected, target)
    actual = split_signal(on, off)
    scores = {}
    for key, desired in wanted.items():
        energy = np.square(desired).sum()
        projection = float((desired * actual[key]).sum() / energy) if energy > 0 else None
        scores[key] = {"projection": projection, "component_mae": float(np.abs(desired - actual[key]).mean()),
                       "passed": projection is not None and .75 <= projection <= 1.25}
    return scores


def render(engine, target, edited, expanded):
    if expanded:
        content = engine.analysis(target)
        prepared = [PreparedSpatial(engine, appearance_grid(engine.model, engine.api, content,
                                                           engine.analysis(guide)), {})
                    for guide in (target, edited)]
    else:
        prepared = [engine.prepare(target, guide) for guide in (target, edited)]
    return {role: finish_spatial(target, state.apply(target), STRENGTH)
            for role, state in zip(("off", "on"), prepared, strict=True)}


def measure(suite, target, expected, images, protected):
    support = int(protected.sum())
    if suite == "film_b" and support < 256:
        raise ValueError("Insufficient protected-region pixels")
    result = {}
    for tag, start in (("native_off", images["off"]), ("from_target", target)):
        if suite == "background":
            result[tag] = background_metrics(target, expected, start, images["on"])
        else:
            result[tag] = {name: metrics(target, expected, start, images["on"], area)
                           for name, area in (("whole", np.ones(protected.shape, dtype=bool)),
                                              ("neutral", protected))}
    result["protected_pixels"] = support
    for role, truth in (("off", target), ("on", expected)):
        result[f"{role}_protected_mae_rgb8"] = (
            float(np.abs(images[role][protected].astype(float) - truth[protected]).mean())
            if support >= 256 else None
        )
    return result


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    parent = verified_json(BASE / FULL / "results.json", FULL_SHA)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Expanded checkpoint changed")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    out.mkdir()
    cases = list(build_cases())
    results = {"background": {}, "film_b": {}}
    sheets = {}
    for arm in ("13", "expanded"):
        if arm == "expanded":
            initialize_replication_model(engine.model, engine.model.state_dict())
            engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
            engine.model.eval().requires_grad_(False)
        for suite, name, target, edited, expected, protected in cases:
            rendered = render(engine, target, edited, arm == "expanded")
            row = results[suite].setdefault(name, {})
            prefix = f"{suite}-{name}"
            row[arm] = measure(suite, target, expected, rendered, protected)
            row[arm]["images"] = {role: save(out / f"{prefix}-{arm}-{role}.png", image)
                                  for role, image in rendered.items()}
            sheets.setdefault(prefix, {}).update({f"{arm} {role}": {"target": target, "reference": guide,
                                                 "result": rendered[role], "expected": wanted}
                for role, guide, wanted in (("off", target, target), ("on", edited, expected))})
    for suite, name, *_ in cases:
        prefix = f"{suite}-{name}"
        row = results[suite][name]
        row["panel_sha256"] = panel(out / f"{prefix}-comparison.png", sheets[prefix])
        cross = parent["suites"][suite]["cases"]
        row["cross_content_expanded"] = (cross[name]["candidate"]["projection"] if suite == "background"
                                          else {key: item["candidate"] for key, item in cross.items()
                                                if key.startswith(f"{name}<-")})
    json_to(out / "results.json", {"parent_sha256": FULL_SHA, "13_sha256": CHECKPOINT,
                                   "expanded_sha256": WEIGHT_SHA256, "strength": STRENGTH, "suites": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
