"""Transfer author-provided desaturation evidence across frozen visual features."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from dense_anchor_model import DenseAnchorModel
from probe_automatic_selective_chroma import PARENT_SHA, render, save
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_psbench_background_desaturation import REC709, TASKS, load, luminance_chroma, panel, regions, score, sha


NEIGHBORS = 5


def descriptors(encoder, image):
    gray = np.rint(image @ REC709).astype(np.uint8)
    data, _, grid, _ = encoder.extract(np.repeat(gray[..., None], 3, axis=2))
    return F.normalize(data, dim=1), grid


def edit_weight(reference, authored, reference_features, target_features, target_shape):
    features, grid = reference_features
    samples, target_grid = target_features
    counts, classes = {}, {}
    for name, area in regions(reference, authored).items():
        pixels = torch.from_numpy(area.astype(np.float32)).to(features.device)
        fractions = F.adaptive_avg_pool2d(pixels[None, None], grid).flatten()
        classes[name] = features[fractions > .5]
        counts[name] = len(classes[name])
    if min(counts.values()) < NEIGHBORS:
        return np.zeros(target_shape), counts

    similarities = {name: (samples @ tokens.T).topk(NEIGHBORS, dim=1).values.mean(dim=1)
                    for name, tokens in classes.items()}
    weight = (similarities["desaturated"] > similarities["retained"]).float().reshape(target_grid)
    weight = F.interpolate(weight[None, None], size=target_shape, mode="bilinear", align_corners=False)
    return weight[0, 0].clamp(0, 1).cpu().numpy(), counts


def run(base, out):
    parent = base / "psbench-background-desaturation-v1"
    earlier = verified_json(parent / "results.json", PARENT_SHA)
    source = base / "psbench-background-desaturation-assets-v1"
    pairs = {}
    for task in TASKS:
        if not earlier["checks"][task]["eligible"]:
            raise ValueError(f"Previously ineligible image pair: {task}")
        names = [f"{task}_{role}.jpg" for role in ("start", "end")]
        pairs[task] = tuple(load(source, name, earlier["input_files"][name])[0] for name in names)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = DenseAnchorModel()
    encoded = {task: descriptors(encoder, original) for task, (original, _) in pairs.items()}
    out.mkdir()
    results = {}
    for task, donor in (TASKS, TASKS[::-1]):
        target, target_truth = pairs[task]
        reference, edited = pairs[donor]
        weight, counts = edit_weight(reference, edited, encoded[donor], encoded[task], target.shape[:2])
        absent, _ = edit_weight(reference, reference, encoded[donor], encoded[task], target.shape[:2])
        renders = {}
        for arm, response in (("global", np.ones_like(weight)), ("matched", weight), ("unchanged", absent)):
            image, limits = render(target, target, 1 - response)
            renders[arm] = (image, limits | {"sha256": save(out / f"{task}-{arm}.png", image)})

        areas = regions(target, target_truth)
        metrics = {arm: score(target, target_truth, target, image, areas) | limits
                   for arm, (image, limits) in renders.items()}
        previous = earlier["results"][task]["regions"]
        new = metrics["matched"]["regions"]
        transfer = (new["desaturated"]["absolute_chroma_errors"]["on"]
                    < previous["desaturated"]["absolute_chroma_errors"]["on"]
                    and new["retained"]["absolute_chroma_errors"]["on"]
                    <= previous["retained"]["absolute_chroma_errors"]["on"] + 1.5)
        negative = bool(np.array_equal(target, renders["unchanged"][0]))
        old_file = parent / f"{task}-on.png"
        if sha(old_file) != earlier["results"][task]["hashes"]["on"]:
            raise ValueError("Baseline identity mismatch")
        with Image.open(old_file) as baseline:
            board = panel(out / f"{task}-comparison.png",
                          {"Target": target, "Author output": target_truth, "13 (edited reference)": np.array(baseline),
                           "Global desaturation": renders["global"][0], "Matched desaturation": renders["matched"][0]}, areas)
        results[task] = {
            "donor": donor, "reference_grid": list(encoded[donor][1]), "reference_cells": counts,
            "weights": {name: {"mean": float(weight[area].mean()), "majority_fraction": float((weight[area] > .5).mean())}
                        for name, area in areas.items()},
            "whole_chroma_mae": {arm: float(np.abs(luminance_chroma(image.astype(float) - target_truth)[1]).mean())
                                 for arm, (image, _) in renders.items()},
            "metrics": metrics, "absolute_gate": transfer, "unchanged_exact": negative, "panel_sha256": board,
            "passed": transfer and negative and metrics["matched"]["passed"],
        }
    json_to(out / "results.json", {"parent_sha256": PARENT_SHA, "encoder": encoder.provenance,
                                   "cases": results, "passed": all(row["passed"] for row in results.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
