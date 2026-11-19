"""Try locally flat RGB residuals as a closed-set style cue."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import REC709, evaluate, fit, natural_scenes, renderings, samples, style_transforms
from probe_frozen_candidate_development_crosscheck import BASE
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json

PARENT = "capture-invariant-look-v1"
PARENT_SHA = "4ca6ff449f9e92cadd9e5560fe97a57809447e279fdf071983bb2ed9cc3e62ff"
SUBS = ("original", "jpeg85")


def checkerboard(tiles):
    return (tiles[:, ::2, ::2] - tiles[:, ::2, 1::2]
            - tiles[:, 1::2, ::2] + tiles[:, 1::2, 1::2])


def descriptor(rgb):
    height, width = rgb.shape[:2]
    rows, cols = height // 8, width // 8
    if rows * cols < 48:
        raise ValueError("Too few complete reference tiles")
    tiles = rgb[:rows * 8, :cols * 8].astype(np.float64).reshape(rows, 8, cols, 8, 3)
    tiles = tiles.transpose(0, 2, 1, 3, 4).reshape(-1, 8, 8, 3)
    light = tiles @ REC709
    block_light = light.mean(axis=(1, 2))
    quadrants = light.reshape(-1, 2, 4, 2, 4).mean(axis=(2, 4))
    structure = quadrants.var(axis=(1, 2))
    residual = checkerboard(tiles)
    luminance = residual @ REC709
    values = np.stack((luminance, residual[..., 0] - luminance, residual[..., 2] - luminance), axis=-1)
    groups = np.array_split(np.argsort(block_light, kind="stable"), 4)
    features = []
    for indices in groups:
        ids = indices[np.argsort(structure[indices], kind="stable")[:len(indices) // 3]]
        samples_from_band = values[ids].reshape(-1, 3)
        centered = samples_from_band - samples_from_band.mean(axis=0)
        covariance = centered.T @ centered / len(centered)
        variance = covariance.diagonal() + .25
        features.extend(np.log(variance / variance.sum()))
        features.extend(covariance[i, j] / np.sqrt(variance[i] * variance[j])
                        for i, j in ((0, 1), (0, 2), (1, 2)))
    result = np.array(features)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite flat-tile description")
    return result


def jpeg_descriptor(rgb):
    with BytesIO() as stream:
        Image.fromarray(rgb).save(stream, format="JPEG", quality=85, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as image:
            return descriptor(np.asarray(image.convert("RGB")))


def run(out):
    historical = verified_json(BASE / PARENT / "results.json", PARENT_SHA)
    reference_history = historical["arms"]["paired"]
    transforms = style_transforms(BASE)
    scenes = natural_scenes(BASE)
    original = samples(BASE, scenes, transforms, descriptor)
    jpeg = samples(BASE, scenes, transforms, jpeg_descriptor)
    outputs = renderings(scenes, transforms)
    if set(original) != set(reference_history) or original.keys() != jpeg.keys() or historical["sources"] != 24:
        raise ValueError("Frozen source groups changed")
    out.mkdir()
    results = {key: {} for key in SUBS}
    for domain in original:
        model = fit([group for source, group in original.items() if source != domain], 1.)
        for role, cohorts in zip(SUBS, (original, jpeg), strict=True):
            folder = out / role / domain
            folder.mkdir(parents=True)
            row = evaluate(cohorts[domain], domain, model, outputs, folder)
            if row["states"] != reference_history[domain]["states"]:
                raise ValueError("Frozen case count changed")
            results[role][domain] = {**row, "train_states": model[-2], "train_differences": model[-1]}
    passes = {role: sum(row["passed"] for row in groups.values()) for role, groups in results.items()}
    json_to(out / "results.json", {"historical_sha256": PARENT_SHA, "arms": results, "passing_domains": passes,
                                   "passed": min(passes.values()) >= 3 and passes["original"] > historical["passing_domains"]["paired"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
