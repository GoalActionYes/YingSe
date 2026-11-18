"""Measure color coverage when the paired reference is supplied offline."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from background_intervention import target_regions
from probe_arri_sdr_renderability import PHOTOS, REPORT_SHA, digest, input_photo, metrics
from spatial_render import finish_spatial

RADIUS = 6.
NEIGHBORS = 8


def mapping(before, after):
    pixels = before.reshape(-1, 3)
    rgb, ids, counts = np.unique(pixels, return_inverse=True, return_counts=True, axis=0)
    if len(rgb) < NEIGHBORS:
        raise ValueError("Insufficient reference colors")
    delta = after.reshape(-1, 3).astype(np.int16) - pixels
    stops = np.cumsum(counts)
    lower, upper = stops - (counts + 2) // 2, stops - (counts + 1) // 2
    values = []
    for channel in delta.T:
        ordered = channel[np.lexsort((channel, ids))]
        values.append((ordered[lower] + ordered[upper]) * .5)
    return rgb, np.stack(values, axis=1)


def estimate(reference_rgb, reference_delta, target):
    palette, inverse = np.unique(target.reshape(-1, 3), return_inverse=True, axis=0)
    distances, indices = cKDTree(reference_rgb).query(palette, k=NEIGHBORS, workers=1)
    weights = 1 / np.maximum(distances, 1)
    correction = np.sum(weights[..., None] * reference_delta[indices], axis=1) / weights.sum(axis=1)[:, None]
    estimated = np.rint(np.clip(palette + correction, 0, 255)).astype(np.uint8)
    shape = target.shape
    return estimated[inverse].reshape(shape), distances[inverse, 0].reshape(shape[:2])


def report_region(original, expected, output, area, supported):
    coverage = supported[area]
    if not coverage.size:
        raise ValueError("Required region is empty")
    error = np.abs(output.astype(np.int16) - expected)[area]
    return {**metrics(original, expected, original, output, area),
            "pixels": int(area.sum()), "coverage": float(coverage.mean()),
            "supported_mae": float(error[coverage].mean()) if coverage.any() else None,
            "unsupported_mae": float(error[~coverage].mean()) if (~coverage).any() else None}


def run(folder, output):
    parent = folder / "results.json"
    if digest(parent) != REPORT_SHA:
        raise ValueError("Frozen Film A source report changed")
    state = json.loads(parent.read_bytes())
    images = {scene: [input_photo(folder, f"{scene}-{role}.png", state["prepared"][scene]["images"][role])
                      for role in ("off", "on")] for scene in PHOTOS}
    output.mkdir()
    cases = {}
    for target, reference in (PHOTOS, PHOTOS[::-1]):
        original, gold = images[target]
        palette, delta = mapping(*images[reference])
        estimate_rgb, distance = estimate(palette, delta, original)
        supported = distance <= RADIUS
        arms = {"supported": np.where(supported[..., None], estimate_rgb, original), "extend": estimate_rgb}
        source = np.ones(original.shape[:2], dtype=bool)
        both_neutral = np.logical_and(*(target_regions(rgb, source)["neutral_proxy"] for rgb in images[target]))
        if int(both_neutral.sum()) != state["prepared"][target]["shared_neutral_pixels"]:
            raise ValueError("Frozen neutral region disagrees")
        expected = finish_spatial(original, gold, .8)
        details = {
            "reference": reference, "reference_colors": len(palette),
            "distance_percentiles_50_80_95": np.percentile(distance, [50, 80, 95]).tolist(), "arms": {},
        }
        for arm, image in arms.items():
            applied = finish_spatial(original, image, .8)
            file = output / f"{target}-{arm}.png"
            Image.fromarray(applied).save(file)
            details["arms"][arm] = {
                "sha256": digest(file),
                "whole": report_region(original, expected, applied, source, supported),
                "shared_neutral": report_region(original, expected, applied, both_neutral, supported),
            }
        cases[target] = details
    passed = all(item["passed"] and item["coverage"] >= .8 for row in cases.values()
                 for name, item in row["arms"]["supported"].items() if name != "sha256")
    result = {"source_sha256": REPORT_SHA, "radius": RADIUS, "neighbors": NEIGHBORS,
              "passed": passed, "cases": cases}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
