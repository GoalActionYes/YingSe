"""Test a fixed compensation for target chroma that disappears under the reference."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_hdm_hdr_display import json_to
from probe_target_mirror_chroma import BG_SHA, FACD_SHA, background, facd
from probe_target_mirror_chroma import CHECKPOINT, LUMA, LUMA_INTEGER, STRENGTH, chroma, render
from spatial_render import SpatialEngine, finish_spatial

TARGET_CHROMA = .75


def chroma_delta(before, after):
    change = before.astype(np.float64) - after
    return chroma(change, change @ LUMA)


def counterfactual(target):
    bright = target @ LUMA
    shifted, _ = render(TARGET_CHROMA * chroma(target, bright), bright)
    return shifted


def recover(target, shifted_target, baseline, shifted_render):
    source_delta = chroma_delta(target, shifted_target)
    grade_delta = chroma_delta(baseline, shifted_render)
    residual = source_delta - grade_delta
    y = baseline @ LUMA
    result, compressed = render(chroma(baseline, y) + residual, y)
    luma_error = np.abs((result.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000
    power = np.sum(source_delta**2)
    return result, {
        "target_change_chroma_mae": float(np.abs(source_delta).mean()),
        "model_change_chroma_mae": float(np.abs(grade_delta).mean()),
        "chroma_projection": float(np.sum(source_delta * grade_delta) / power) if power else None,
        "correction_chroma_mae": float(np.abs(residual).mean()),
        "output_change_mae": float(np.abs(result.astype(np.int16) - baseline).mean()),
        "max_luma_error": float(luma_error),
        "gamut_fraction": compressed,
    }


def compensate(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True}
    quiet = counterfactual(target)
    graded = finish_spatial(quiet, engine.prepare(quiet, reference).apply(quiet), STRENGTH)
    return recover(target, quiet, baseline, graded)


def check_formula():
    colors = np.array([[[205, 132, 92], [72, 96, 156], [142, 142, 142]]], dtype=np.uint8)
    changed = counterfactual(colors)
    assert np.max(np.abs((colors.astype(np.int32) - changed) @ LUMA_INTEGER)) <= 5000
    identity, exact = recover(colors, changed, colors, changed)
    np.testing.assert_array_equal(identity, colors)
    assert exact["correction_chroma_mae"] == 0
    fixed = np.full_like(colors, 142)
    corrected, erased = recover(colors, changed, fixed, fixed)
    np.testing.assert_allclose(chroma_delta(corrected, fixed), chroma_delta(colors, changed), atol=1.)
    assert erased["chroma_projection"] == 0
    same, _ = compensate(None, colors, colors.copy(), 255 - colors)
    np.testing.assert_array_equal(same, colors)


def run(output):
    check_formula()
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    output.mkdir()
    real = background(engine, output / "bg-real", compensate, "chroma compensation")
    look = facd(engine, output / "facd", compensate, "chroma compensation")
    max_y_error = max(row["max_luma_error"] for block in (real, look)
                      for case in block["cases"].values() for row in case["outputs"].values())
    json_to(output / "results.json", {
        "checkpoint_sha256": CHECKPOINT, "strength": STRENGTH, "input_chroma_factor": TARGET_CHROMA,
        "background_baseline_sha256": BG_SHA, "facd_baseline_sha256": FACD_SHA,
        "background": real, "facd": look, "max_luma_error": max_y_error,
        "passed": real["passed"] and look["passed"] and max_y_error <= .5,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
