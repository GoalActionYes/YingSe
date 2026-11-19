"""Split local desaturation evidence by pixel hue before matching it."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dense_anchor_model import DenseAnchorModel
from probe_automatic_selective_chroma import PARENT_SHA, controls
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_paired_semantic_desaturation import descriptors
from probe_psbench_background_desaturation import TASKS, load, regions
from probe_psbench_edge_guided_mask import oracle_grid, record
from probe_psbench_selective_fractional import TEMP, apply, automatic, labels
from probe_selective_hue_support import BINS, angles, hue_support
from probe_target_mirror_chroma import checked_rgb, panel

FROZEN = "a8e9cd7f51afc9af2197731f41ea91e33f1f9a8ef9aa568f3ee25004329cd23e"


def pool(layers, grid, device):
    data = torch.as_tensor(layers, dtype=torch.float32, device=device)
    return F.adaptive_avg_pool2d(data[None], grid).flatten(2)[0].T.contiguous()


def layers(image, area):
    buckets = np.arange(BINS)[:, None, None]
    return (angles(image).astype(np.int32)[None] == buckets) & area[None]


def upsample(votes, grid, shape):
    maps = votes.T.reshape(1, -1, *grid)
    return F.interpolate(maps, shape, mode="bilinear", align_corners=False)[0].cpu().numpy()


def at_hue(maps, image):
    position = angles(image) - .5
    bucket = np.floor(position).astype(np.int32)
    fraction = position - bucket
    rows, cols = np.indices(position.shape)
    return maps[bucket % BINS, rows, cols] * (1 - fraction) + maps[(bucket + 1) % BINS, rows, cols] * fraction


def localized(encoder, image, reference, encoded):
    shape = image.shape[:2]
    if np.array_equal(image, reference) or not hue_support(reference)[1]["applied"]:
        return np.zeros(shape), np.zeros(shape), {"active": False}

    keys, source_grid = descriptors(encoder, reference)
    queries, target_grid = encoded
    candidate = labels(reference)
    colored = pool(layers(reference, candidate["retained"]), source_grid, keys.device)
    gray = pool(candidate["desaturated"][None], source_grid, keys.device)
    gray_area, color_area = gray.sum().item(), colored.sum().item()
    detail = {"gray_area": gray_area, "color_area": color_area, "active": False}
    if min(gray_area, color_area) < 5:
        return np.zeros(shape), np.zeros(shape), detail

    distances = torch.arange(BINS, device=keys.device, dtype=keys.dtype)
    similarity = torch.cos((distances[:, None] - distances) * (2 * np.pi / BINS)).clamp(min=0).square()
    support = colored @ similarity
    kernel = torch.exp((queries @ keys.T - 1) / TEMP)
    retain = (kernel @ support) / support.sum(dim=0).clamp(min=1)
    remove = (kernel @ gray) / gray_area
    keep_at_pixel = at_hue(upsample(retain, target_grid, shape), image)
    remove_at_pixel = upsample(remove, target_grid, shape)[0]
    weight = (remove_at_pixel > keep_at_pixel).astype(np.float32)
    unconditioned = (kernel @ colored.sum(dim=1, keepdim=True)) / color_area
    control = (remove_at_pixel > upsample(unconditioned, target_grid, shape)[0]).astype(np.float32)
    detail.update(active=bool(weight.any()), support=support.sum(dim=0).cpu().tolist())
    return weight, control, detail


def known_local(image, areas, grid):
    all_labelled = areas["desaturated"] | areas["retained"]
    cpu = torch.device("cpu")
    selected = pool(layers(image, areas["desaturated"]), grid, cpu)
    labelled = pool(layers(image, all_labelled), grid, cpu)
    return at_hue(upsample(selected / labelled.clamp(min=1e-9), grid, image.shape[:2]), image)


@torch.inference_mode()
def run(base, out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    upstream = base / "psbench-edge-guided-v1"
    frozen = verified_json(upstream / "results.json", FROZEN)
    originals = verified_json(base / "psbench-background-desaturation-v1/results.json", PARENT_SHA)
    media = base / "psbench-background-desaturation-assets-v1"
    pairs = {task: [load(media, f"{task}_{s}.jpg", originals["input_files"][f"{task}_{s}.jpg"])[0]
                    for s in ("start", "end")] for task in TASKS}
    encoder = DenseAnchorModel()
    out.mkdir()
    cases, negatives = {}, {}
    for task, donor in (TASKS, TASKS[::-1]):
        image, truth = pairs[task]
        reference_before, reference = pairs[donor]
        encoded = descriptors(encoder, image)
        grid, shape = encoded[1], image.shape[:2]
        old_weight, _ = automatic(encoder, image, reference, encoded)
        hue_weight, control, detail = localized(encoder, image, reference, encoded)
        areas = regions(image, truth)
        weights = {"old": old_weight, "pixel_vote": control, "local_hue": hue_weight,
                   "oracle_grid": oracle_grid(areas, grid, shape), "oracle_hue": known_local(image, areas, grid)}
        board, evidence = {"target": image, "reference": reference, "author": truth}, {}
        for arm, weight in weights.items():
            board[arm], evidence[arm] = record(out, task, arm, image, truth, areas, weight)
        for arm in ("old", "oracle_grid"):
            prior = checked_rgb(upstream / f"{task}-{arm}.png", frozen["cases"][task]["arms"][arm]["sha256"])
            np.testing.assert_array_equal(board[arm], prior)
        cases[task] = {"grid": list(grid), "arms": evidence, "detail": detail,
                       "comparison_sha256": panel(out / f"{task}-comparison.png", {"Color within cells": board})}
        others = [(f"{task}-identity", image), (f"{task}-unedited", reference_before)]
        if task == TASKS[0]:
            others.extend(controls(base, pairs))
        for key, other in others:
            weight, _, _ = localized(encoder, image, other, encoded)
            negatives[key] = bool(np.array_equal(apply(image, weight), image))
    assert len(negatives) == 12
    passed = all(case["arms"]["local_hue"]["scores"]["passed"] for case in cases.values()) and all(negatives.values())
    json_to(out / "results.json", {"previous_sha256": FROZEN, "encoder": encoder.provenance,
                                   "cases": cases, "negatives": negatives, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
