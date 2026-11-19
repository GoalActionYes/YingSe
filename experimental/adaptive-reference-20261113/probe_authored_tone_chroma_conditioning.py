"""Test whether reference midpoint chroma explains within-tone edit differences."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe_authored_tone_curve_replay import BASE, CENTERS, LUMA, get_color, json_to, load, measured, midrank, render_shifts, run

PARENT = "9c79c6a77de82bd79d295ec07e4cf03a5805ca82c3a9c955f273fb001649e57c"
ARMS = ("curve", "conditioned")


def residual_nodes(reference, midpoint, profile, target_color):
    rank = midrank(midpoint @ LUMA)
    assignment = np.minimum((rank * len(CENTERS)).astype(int), len(CENTERS) - 1)
    chroma = get_color(midpoint)
    difference = .8 * (get_color(reference[1]) - get_color(reference[0]))
    residuals, supported = [], []
    for i, nominal in enumerate(profile):
        selected = assignment == i
        samples = chroma[selected]
        center = np.median(samples, axis=0)
        design = (samples - center) / 32
        residual = difference[selected] - nominal
        penalty = .05 * len(samples) * np.eye(2)
        matrix = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
        lower, upper = np.quantile(samples, [.05, .95], axis=0)
        active = ((target_color >= lower) & (target_color <= upper)).all(axis=-1)
        residuals.append(np.where(active[..., None], ((target_color - center) / 32) @ matrix, 0))
        supported.append(active)
    return np.stack(residuals, axis=2), np.stack(supported, axis=2)


def predict(target, reference, baseline):
    profile, counts = measured(reference)
    if profile is None:
        return {name: [image.copy() for image in baseline] for name in ARMS}, {"applied": False, "counts": counts}
    rank = midrank(target @ LUMA)
    mean = profile.mean(axis=0)
    curve = np.stack([np.interp(rank, CENTERS, profile[:, index]) for index in range(2)], axis=2)
    curve += mean - curve.mean(axis=(0, 1))
    ref_midpoint = .5 * (reference[0].astype(float) + reference[1])
    target_mid_chroma = .5 * (get_color(baseline[0]) + get_color(baseline[1]))
    nodes, supported = residual_nodes(reference, ref_midpoint, profile, target_mid_chroma)
    indices = np.searchsorted(CENTERS, rank, side="right")
    first = np.clip(indices - 1, 0, len(CENTERS) - 1)
    second = np.clip(indices, 0, len(CENTERS) - 1)
    ratio = np.clip((rank - CENTERS[first]) / .2, 0, 1)
    yy, xx = np.indices(rank.shape)
    shift = nodes[yy, xx, first] * (1 - ratio[..., None]) + nodes[yy, xx, second] * ratio[..., None]
    confidence = supported[yy, xx, first] * (1 - ratio) + supported[yy, xx, second] * ratio
    if confidence.any():
        shift -= confidence[..., None] * (shift.sum(axis=(0, 1)) / confidence.sum())
    candidate = curve + shift
    np.testing.assert_allclose(candidate.mean(axis=(0, 1)), mean, rtol=0, atol=1e-10)
    outputs, gamut = render_shifts(baseline, {"curve": curve, "conditioned": candidate})
    return outputs, {"applied": True, "counts": counts, "gamut_fraction": gamut,
                     "supported_fraction": float((confidence > 0).mean()),
                     "effective_support_fraction": float(confidence.mean()),
                     "added_color_mae": float(np.abs(shift).mean())}


def evaluate(out):
    frozen = load(BASE.parent / "authored-tone-curve-replay-v1/results.json", PARENT)
    run(out, predict=predict, arms=ARMS)
    path = out / "results.json"
    report = json.loads(path.read_bytes())
    for key, group in report["cases"].items():
        for slot, frame in group.items():
            if frame["arms"]["curve"] != frozen["cases"][key][slot]["arms"]["curve"]:
                raise ValueError("Frozen tone-curve control changed")
    report["curve_parent_sha256"] = PARENT
    json_to(path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    evaluate(parser.parse_args().out)
