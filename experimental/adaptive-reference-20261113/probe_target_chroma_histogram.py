"""Ablate only the target color histogram and preserve 13's rendered luma."""

import argparse
from pathlib import Path

import numpy as np
import torch

from probe_facd_reference_factorial import ORDER, panel, score
from probe_facd_two_image_baseline import CHECKPOINT, read_checked
from probe_hdm_hdr_display import json_to
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_target_mirror_chroma import (
    BASE, ROOT, FACD_SOURCE_SHA, LUMA_INTEGER, STRENGTH, background, checked_json,
    checked_rgb, facd, save,
)
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output


FACTORIAL_SHA = "b5c381bf6c619c66b27a5ee938897505f8d98b7c1fa0d174c830f123c92a77a9"


@torch.inference_mode()
def prepare(engine, target, reference, *, neutral_target, histogram_image=None):
    api = engine.api
    content, guide = engine.analysis(target), engine.analysis(reference)
    stats_input = content if histogram_image is None else engine.analysis(histogram_image)
    target_stats = api["srgb_to_ycbcr"](stats_input)
    original_y = target_stats[:, :1].clone()
    if neutral_target:
        target_stats[:, 1:] = 0
    torch.testing.assert_close(target_stats[:, :1], original_y, rtol=0, atol=0)
    target_histogram = api["batch_histogram_1D"](target_stats)
    reference_histogram = api["batch_histogram_1D"](api["srgb_to_ycbcr"](guide))
    low = api["TF"].resize(content, (64, 64), antialias=True)
    grids = api["lowpass"](engine.model.apply_grid(low, torch.cat([target_histogram, reference_histogram], 1)))
    finite_output(grids, (1, 13, 8, 16, 16))
    return PreparedSpatial(engine, grids.detach().clone(), {})


def candidate(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"max_luma_error": 0., "identity": True}
    prepared = prepare(engine, target, reference, neutral_target=True)
    new = finish_spatial(target, prepared.apply(target), STRENGTH)
    baseline_y = baseline @ LUMA
    output, limited = render(chroma(new, new @ LUMA), baseline_y)
    error = np.abs((output.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000
    return output, {"max_luma_error": float(error), "gamut_limited_fraction": limited}


def factorial(engine, out):
    path = BASE / "facd-reference-factorial-joint-v1"
    frozen = path / "results.json"
    records = checked_json(frozen, FACTORIAL_SHA)["cases"]
    aligned = BASE / "facd-border-registration-v1"
    source = checked_json(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json", FACD_SOURCE_SHA)
    output = {}
    for name in ("64573", "379484"):
        style = "Hudson"
        previous = records[name]["styles"][style]
        folder = path / name / style
        dest = out / name
        dest.mkdir(parents=True)
        target = read_checked(aligned, name, "Origin", source)
        guides, baselines, grades, audit = {}, {}, {}, {}
        for role in ORDER:
            guides[role] = checked_rgb(folder / f"reference-{role}.png", previous["reference_sha256"][role])
            baselines[role] = checked_rgb(folder / f"grade-{role}.png", previous["output_sha256"][role])
            direct = prepare(engine, target, guides[role], neutral_target=False)
            native = engine.prepare(target, guides[role])
            torch.testing.assert_close(direct.grid, native.grid, rtol=0, atol=0)
            np.testing.assert_array_equal(finish_spatial(target, native.apply(target), STRENGTH), baselines[role])
            y = baselines[role] @ LUMA
            np.testing.assert_array_equal(render(chroma(baselines[role], y), y)[0], baselines[role])
            grades[role], audit[role] = candidate(engine, target, guides[role], baselines[role])
            audit[role]["sha256"] = save(dest / f"grade-{role}.png", grades[role])
        expected = finish_spatial(target, read_checked(aligned, name, style, source), STRENGTH)
        original = score(target, expected, baselines)
        if original != previous["target_score"]:
            raise ValueError("Unexpected factorial baseline score")
        output[name] = {
            "style": style, "baseline": original, "candidate": score(target, expected, grades),
            "outputs": audit,
            "comparison_sha256": panel(dest, guides, grades, target, expected, "target ablation (80%)"),
        }
    return output


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    toy = np.array([[[0, 10, 100], [128, 129, 240]]], dtype=np.uint8)
    np.testing.assert_array_equal(candidate(engine, toy, toy.copy(), 255-toy)[0], toy)
    out.mkdir()
    direct = factorial(engine, out / "factorial")
    bg = background(engine, out / "bg-real", candidate, "target ablation")
    authored = facd(engine, out / "facd", candidate, "target ablation")
    cases = [*direct.values(), *bg["cases"].values(), *authored["cases"].values()]
    max_y = max(row["max_luma_error"] for case in cases for row in case["outputs"].values())
    json_to(out / "results.json", {
        "weight_sha256": CHECKPOINT, "strength": STRENGTH,
        "factorial_sha256": FACTORIAL_SHA, "factorial": direct, "background": bg, "facd": authored,
        "max_luma_error": max_y, "passed": bg["passed"] and authored["passed"] and max_y <= .5,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
