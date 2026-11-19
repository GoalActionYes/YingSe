"""Run the fixed expanded checkpoint through all four photo-development suites."""

import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import digest, metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_hdm_hdr_display import json_to
from probe_stable_chroma_hybrid import chart
from probe_target_mirror_chroma import BASE, STRENGTH, background, facd, panel, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import MANIFESTS, manifest, verify


LABEL = "Expanded epoch 4"
COUNTS = {"background": 40, "facd": 70, "film_b": 18, "chart": 4}


def grade(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True}
    grid = appearance_grid(engine.model, engine.api, engine.analysis(target), engine.analysis(reference))
    result = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)
    return result, {"change_from_13_rgb8": float(np.abs(result.astype(np.int16) - baseline).mean())}


def film_b(engine, out):
    source = "arri-two-looks-v2"
    crossed = "arri-film-b-crossed-v1"
    preparation = manifest(BASE, source)
    baseline = manifest(BASE, crossed)
    if baseline["checkpoint_sha256"] != CHECKPOINT or baseline["source_sha256"] != MANIFESTS[source]:
        raise ValueError("Film B baseline changed")
    scenes = {scene: endpoints(BASE / source, preparation, scene) for scene in ASSIGNMENTS}
    cases = {}
    out.mkdir()
    for name, row in baseline["cases"].items():
        scene, donor = row["target"], row["reference"]
        original, styled = scenes[scene]
        expected = finish_spatial(original, styled, STRENGTH)
        whole = np.ones(original.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(image, ~whole)["neutral_proxy"] for image in (original, styled)))
        if neutral.sum() < 256:
            raise ValueError("Lost frozen neutral region")
        prefix = f"{scene}-from-{donor}"
        outputs, artifacts, comparisons = [], {}, {}
        for state, reference in zip(("off", "film_b"), scenes[donor], strict=True):
            tag = f"{prefix}-{state}.png"
            base_image = verified(BASE / crossed / tag, row["output_sha256"][state])
            result, change = grade(engine, original, reference, base_image)
            outputs.append(result)
            artifacts[state] = change | {"sha256": save(out / tag, result)}
            comparisons[state] = {"target": original, "reference": reference, "13": base_image, LABEL: result,
                                  "expected": expected if state == "film_b" else original}
        measurements = {region: metrics(original, expected, *outputs, mask)
                        for region, mask in (("whole", whole), ("neutral", neutral))}
        cases[name] = {"baseline": {region: row[region] for region in measurements}, "candidate": measurements,
                       "outputs": artifacts, "passed": all(value["passed"] for value in measurements.values()),
                       "comparison_sha256": panel(out / f"{prefix}-comparison.png", comparisons)}
    count = sum(row["passed"] for row in cases.values())
    return {"cases": cases, "passing_cases": count, "passed": count == 6}


@torch.inference_mode()
def run(out):
    checked = verify(BASE, COUNTS)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Expanded checkpoint changed")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    initialize_replication_model(engine.model, engine.model.state_dict())
    state = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Nonfinite expanded weights")
    engine.model.load_state_dict(state, strict=True)
    engine.model.eval().requires_grad_(False)
    sample = np.array([[[4, 20, 200], [255, 0, 128]]], dtype=np.uint8)
    same, _ = grade(engine, sample, sample.copy(), 255 - sample)
    np.testing.assert_array_equal(same, sample)
    out.mkdir()
    suites = {
        "background": background(engine, out / "background", grade, LABEL),
        "facd": facd(engine, out / "facd", grade, LABEL),
        "film_b": film_b(engine, out / "film_b"),
        "chart": chart(engine, out / "chart", grade, LABEL),
    }
    json_to(out / "results.json", {
        "baseline_sha256": CHECKPOINT, "candidate_sha256": WEIGHT_SHA256, "inputs": MANIFESTS,
        "preflight_counts": checked, "strength": STRENGTH, "identity_passed": True, "suites": suites,
        "passed": all(row["passed"] for row in suites.values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
