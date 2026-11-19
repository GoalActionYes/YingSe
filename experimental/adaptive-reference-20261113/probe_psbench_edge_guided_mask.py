"""Test whether aligning a frozen edit mask to target edges reduces spill."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from dense_anchor_model import DenseAnchorModel
from probe_automatic_selective_chroma import PARENT_SHA, controls
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_paired_semantic_desaturation import descriptors
from probe_psbench_background_desaturation import REC709, TASKS, load, regions, score
from probe_psbench_selective_fractional import apply, automatic
from probe_target_mirror_chroma import checked_rgb, panel, save

FROZEN = "f3e9933f74d1b0e15a4f9dc1704fbbbd66b33c37c83623d1c2731c2f9125217a"


def guided(image, weight):
    light = image @ REC709
    def local_mean(values):
        return cv2.boxFilter(values, -1, (15, 15), borderType=cv2.BORDER_REFLECT)
    mean = local_mean(light)
    mean_weight = local_mean(weight.astype(np.float64))
    variance = local_mean(light * light) - mean * mean
    covariance = local_mean(light * weight) - mean * mean_weight
    a = covariance / (np.maximum(variance, 0) + 64)
    b = mean_weight - a * mean
    return np.clip(local_mean(a) * light + local_mean(b), 0, 1)


def oracle_grid(area, grid, shape):
    selected = torch.from_numpy(area["desaturated"].astype(np.float32))[None, None]
    labelled = torch.from_numpy((area["desaturated"] | area["retained"]).astype(np.float32))[None, None]
    support = F.adaptive_avg_pool2d(labelled, grid)
    weight = F.adaptive_avg_pool2d(selected, grid) / support.clamp(min=1e-9)
    return F.interpolate(weight, shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def record(out, task, arm, target, truth, areas, weight):
    result = apply(target, weight)
    peak = float(np.abs((result.astype(float) - target) @ REC709).max())
    if peak > .5 + 1e-9:
        raise ValueError(f"Unexpected luminance change for {task}-{arm}")
    props = {"sha256": save(out / f"{task}-{arm}.png", result), "peak_luma": peak,
             "scores": score(target, truth, target, result, areas),
             "avg_weight": {name: float(weight[area].mean()) for name, area in areas.items()}}
    return result, props


@torch.inference_mode()
def run(base, out):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen_path = base / "psbench-selective-fractional-v1"
    frozen = verified_json(frozen_path / "results.json", FROZEN)
    originals = verified_json(base / "psbench-background-desaturation-v1/results.json", PARENT_SHA)
    media = base / "psbench-background-desaturation-assets-v1"
    pairs = {task: [load(media, f"{task}_{s}.jpg", originals["input_files"][f"{task}_{s}.jpg"])[0]
                    for s in ("start", "end")] for task in TASKS}
    encoder = DenseAnchorModel()
    out.mkdir()
    cases, negatives = {}, {}
    for task, donor in (TASKS, TASKS[::-1]):
        target, truth = pairs[task]
        reference_before, reference = pairs[donor]
        encoded = descriptors(encoder, target)
        old, _ = automatic(encoder, target, reference, encoded)
        areas = regions(target, truth)
        ideal = oracle_grid(areas, encoded[1], target.shape[:2])
        weights = {"old": old, "guided": guided(target, old), "oracle_grid": ideal,
                   "oracle_guided": guided(target, ideal)}
        evidence, board = {}, {"target": target, "reference": reference, "author": truth}
        for arm, weight in weights.items():
            board[arm], evidence[arm] = record(out, task, arm, target, truth, areas, weight)
        archived = checked_rgb(frozen_path / f"{task}-automatic.png",
                               frozen["cases"][task]["arms"]["automatic"]["sha256"])
        np.testing.assert_array_equal(board["old"], archived)
        cases[task] = {"grid": list(encoded[1]), "arms": evidence,
                       "comparison_sha256": panel(out / f"{task}-comparison.png", {"Mask comparison": board})}
        others = [(f"{task}-identity", target), (f"{task}-unedited", reference_before)]
        if task == TASKS[0]:
            others.extend(controls(base, pairs))
        for key, other in others:
            weight, _ = automatic(encoder, target, other, encoded)
            negatives[key] = bool(np.array_equal(apply(target, guided(target, weight)), target))
    assert len(negatives) == 12
    passes = all(case["arms"]["guided"]["scores"]["passed"] for case in cases.values()) and all(negatives.values())
    json_to(out / "results.json", {"parent_sha256": FROZEN, "encoder": encoder.provenance,
                                   "cases": cases, "negatives": negatives, "passed": passes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
