"""Retain minority edits within reference cells when transferring desaturation."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dense_anchor_model import DenseAnchorModel
from probe_automatic_selective_chroma import PARENT_SHA, controls
from probe_hdm_hdr_equal_luminance import verified_json
from probe_hdm_hdr_display import json_to
from probe_paired_semantic_desaturation import descriptors
from probe_psbench_background_desaturation import TASKS, REC709, load, luminance_chroma, regions, score
from probe_selective_hue_support import hue_support
from probe_target_mirror_chroma import panel, save

TEMP = .07


def labels(reference):
    _, chroma = luminance_chroma(reference)
    distance = np.linalg.norm(chroma, axis=-1)
    return {"desaturated": distance <= 3, "retained": distance >= 12}


def estimate(source, target, areas, shape):
    features, grid = source
    queries, shape_small = target
    votes = {}
    support = {}
    kernel = torch.exp((queries @ features.T - 1) / TEMP)
    for name, mask in areas.items():
        layer = torch.from_numpy(mask.astype(np.float32)).to(features.device)
        fractions = F.adaptive_avg_pool2d(layer[None, None], grid).flatten()
        support[name] = {"area": float(fractions.sum().item()), "occupied": int((fractions > 0).sum().item()),
                         "majority": int((fractions > .5).sum().item())}
        votes[name] = (kernel @ fractions) / fractions.sum().clamp(min=1)
    if min(x["area"] for x in support.values()) < 5:
        return np.zeros(shape), support
    pos = (votes["desaturated"] > votes["retained"]).to(dtype=torch.float32).reshape(shape_small)
    w = F.interpolate(pos[None, None], shape, mode="bilinear", align_corners=False)[0, 0]
    return w.cpu().numpy(), support


def apply(source, w):
    light = (source @ REC709)[..., None]
    return np.rint(source.astype(float) + .8*w[..., None]*(light-source)).astype(np.uint8)


def automatic(encoder, src, ref, enc):
    if np.array_equal(src, ref):
        return np.zeros(src.shape[:2]), {"active": False, "identity": True}
    _, guard = hue_support(ref)
    if not guard["applied"]:
        return np.zeros(src.shape[:2]), {"active": False, "guard": guard}
    edit, label_support = estimate(descriptors(encoder, ref), enc, labels(ref), src.shape[:2])
    return edit, {"active": bool(edit.any()), "support": label_support, "guard": guard}


@torch.inference_mode()
def run(base, out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    existing = verified_json(base / "psbench-background-desaturation-v1/results.json", PARENT_SHA)
    media = base / "psbench-background-desaturation-assets-v1"
    pairs = {task: [load(media, f"{task}_{s}.jpg", existing["input_files"][f"{task}_{s}.jpg"])[0]
                    for s in ("start", "end")] for task in TASKS}
    encoder = DenseAnchorModel()
    encoded = {k: descriptors(encoder, v[0]) for k,v in pairs.items()}
    out.mkdir()
    cases, negatives = {}, {}
    for task, donor in (TASKS, TASKS[::-1]):
        original, truth = pairs[task]
        ref_before, ref_after = pairs[donor]
        weight, audit = automatic(encoder, original, ref_after, encoded[task])
        known, known_support = estimate(descriptors(encoder, ref_after), encoded[task],
                                        regions(ref_before, ref_after), original.shape[:2])
        outputs, board = {}, {"target": original, "reference": ref_after, "author": truth}
        selection = regions(original, truth)
        for label, w in (("automatic", weight), ("oracle_reference", known)):
            result = apply(original, w)
            assert np.abs((result.astype(float) - original) @ REC709).max() <= .5 + 1e-9
            outputs[label] = {"scores": score(original, truth, original, result, selection),
                              "weights": {k: float(w[area].mean()) for k,area in selection.items()},
                              "sha256": save(out / f"{task}-{label}.png", result)}
            board[label] = result
        cases[task] = {"arms": outputs, "automatic": audit, "oracle_support": known_support,
                       "comparison_sha256": panel(out / f"{task}-comparison.png", {"reference match": board})}
        other = [(f"{task}-identity", original), (f"{task}-unedited", ref_before)]
        if task == TASKS[0]:
            other.extend(controls(base, pairs))
        for key, ref in other:
            w, info = automatic(encoder, original, ref, encoded[task])
            negatives[key] = info | {"byte_exact": bool(np.array_equal(apply(original, w), original))}
    count = sum(case["arms"]["automatic"]["scores"]["passed"] for case in cases.values())
    json_to(out / "results.json", {"parent_sha256": PARENT_SHA, "encoder": encoder.provenance,
              "cases": cases, "negatives": negatives, "passing_cases": count,
              "passed": count == 2 and all(v["byte_exact"] for v in negatives.values())})
    print("passing", count, "negative", sum(x["byte_exact"] for x in negatives.values()), len(negatives))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
