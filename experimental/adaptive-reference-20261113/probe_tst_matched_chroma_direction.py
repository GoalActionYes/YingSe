"""Compare matched reference chroma with a location-permuted control."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dense_anchor_model import DenseAnchorModel
from probe_paired_semantic_desaturation import NEIGHBORS, descriptors
from probe_tst_neutral_direction_fit import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, checked_rgb,
    get_color, images, json_to, load, masks, measure, neutral, render, save,
    tensor_samples, whole,
)

ARMS = ("matched", "shifted")


def describe(encoder, image):
    tokens, grid = descriptors(encoder, image)
    color = torch.from_numpy(get_color(image)).to(tokens).permute(2, 0, 1)
    return tokens, F.adaptive_avg_pool2d(color, grid)


def directions(target_data, reference_data, shape):
    target_tokens, target_color = target_data
    ref_tokens, ref_color = reference_data
    indices = (target_tokens @ ref_tokens.T).topk(NEIGHBORS, dim=1).indices
    grid = target_color.shape
    shifted = torch.roll(ref_color, shifts=ref_color.shape[1] // 2, dims=1)
    result = {}
    for name, colors in zip(ARMS, (ref_color, shifted), strict=True):
        picked = colors.flatten(1).T[indices].mean(dim=1)
        delta = picked.T.reshape(grid) - target_color
        result[name] = F.interpolate(
            delta[None], size=shape, mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
    return result


def grade(target, reference, baseline, area, target_data=None, reference_data=None):
    if np.array_equal(target, reference):
        return {name: target.copy() for name in ARMS}
    outputs = {name: baseline.copy() for name in ARMS}
    if not area.any():
        return outputs
    original = get_color(target)[area]
    length = np.linalg.norm(get_color(baseline)[area] - original, axis=1)
    light = (baseline @ LUMA)[area]
    for name, delta in directions(target_data, reference_data, area.shape).items():
        vector = delta[area].astype(float)
        norm = np.linalg.norm(vector, axis=1)
        active = norm > 1e-12
        if not active.any():
            continue
        color = original + vector * np.divide(
            length, norm, out=np.zeros_like(length), where=active
        )[:, None]
        pixels = baseline[area].copy()
        pixels[active], _ = render(color[active], light[active])
        outputs[name][area] = pixels
    return outputs


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    ready = {row["index"] for row in assay["cases"] if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    if len(ready) != 5 or len(split["development"]) != 24:
        raise ValueError("Unexpected development cohort")
    encoder = DenseAnchorModel()
    out.mkdir()
    results = {}
    for index, tensors in enumerate(tensor_samples(split["development"])):
        target, refs, truths = images(tensors)
        area = neutral(target)
        base = [checked_rgb(archive / f"{index}-{i}.png", stable["predictions"][f"{index}-{i}.png"])
                for i in range(3)]
        target_data = describe(encoder, target)
        changed = [grade(target, ref, frame, area, target_data, describe(encoder, ref))
                   for ref, frame in zip(refs, base, strict=True)]
        for frame in grade(target, target.copy(), base[0], area).values():
            np.testing.assert_array_equal(frame, target)
        for frame in grade(target, refs[0], base[0], np.zeros_like(area)).values():
            np.testing.assert_array_equal(frame, base[0])
        selected = masks(target, (truths[0], truths[2])) if index in ready else None
        rows = {}
        for name, frames in [("13", base), *[(arm, [item[arm] for item in changed]) for arm in ARMS]]:
            row = {"whole": whole(frames, base, truths)}
            if selected is not None:
                row["local"] = measure(frames, (truths[0], truths[2]), selected)
            if name != "13":
                row["hashes"] = []
                for slot, (before, after) in enumerate(zip(base, frames, strict=True)):
                    np.testing.assert_array_equal(before[~area], after[~area])
                    row["hashes"].append(save(out / f"{index}-{slot}-{name}.png", after))
            elif row["whole"]["passed"] != stable["cases"][index]["passed"]:
                raise ValueError(f"Changed baseline result {index}")
            rows[name] = row
        results[str(index)] = rows
    if len(results) != 24:
        raise ValueError("Development cohort did not fully render")
    summary = {}
    for name in ("13", *ARMS):
        arms = [case[name] for case in results.values()]
        summary[name] = {
            "whole_pass": sum(arm["whole"]["passed"] for arm in arms),
            "local_pass": sum(arm.get("local", {}).get("passed", False) for arm in arms),
            "worst_regression": max(arm["whole"]["worst_regression"] for arm in arms),
        }
    candidate = summary["matched"]
    passed = (candidate["local_pass"] >= 3 and candidate["whole_pass"] >= 7
              and candidate["worst_regression"] <= 2
              and candidate["local_pass"] >= summary["shifted"]["local_pass"] + 2)
    json_to(out / "results.json", {"split_sha256": SPLIT_SHA, "baseline_sha256": RENDER_SHA,
                                  "descriptor": encoder.provenance, "cases": results,
                                  "summary": summary, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
