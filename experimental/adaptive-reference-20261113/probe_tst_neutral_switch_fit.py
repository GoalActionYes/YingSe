"""Train frozen-render chroma residuals with and without switch-pixel weighting."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_tst_neutral_switch_renders import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA,
    checked_rgb, finish_spatial, get_color, json_to, load, masks, measure, render, save,
)
from background_intervention import target_regions
from probe_tst_reference_spatial_signal import structure_score
from train_tst_reference_equivalence import SpatialEngine, tensor_samples


def neutral(rgb):
    return target_regions(rgb, np.zeros(rgb.shape[:2], bool))["neutral_proxy"]


def features(target, guide, output, selection):
    light = (target @ LUMA)[selection] / 255
    original = get_color(target)[selection] / 32
    delta = (get_color(output)[selection] - get_color(target)[selection]) / 32
    # Fixed reference chroma summary and its interaction with target lightness.
    ref = (np.quantile(get_color(guide), [.25, .5, .75], axis=(0, 1))
           - np.quantile(get_color(target), [.25, .5, .75], axis=(0, 1))).reshape(-1) / 64
    return np.column_stack([np.ones(len(light)), light, light**2, original, delta,
                            np.broadcast_to(ref, (len(light), ref.size)), light[:, None] * ref])


def grade(target, guide, baseline, coefficients):
    if np.array_equal(target, guide):
        return target.copy()
    area = neutral(target)
    if not area.any():
        return baseline.copy()
    revised = baseline.copy()
    change = np.clip(features(target, guide, baseline, area) @ coefficients, -8, 8)
    rgb, _ = render(get_color(baseline)[area] + change, (baseline @ LUMA)[area])
    revised[area] = rgb
    return revised


def rgb(tensor):
    return tensor.permute(1, 2, 0).numpy().copy()


def images(tensors):
    target, guides, truths = tensors
    target = rgb(target[0])
    return target, [rgb(x) for x in guides], [finish_spatial(target, rgb(x), .8) for x in truths]


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    cohort = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    cases = {row["index"]: row for row in cohort["cases"]
             if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    assert len(split["training"]) == 82 and len(split["development"]) == 24 and len(cases) == 5
    engine = SpatialEngine("multiref-v1")
    assert stable["checkpoint_sha256"] == engine.checkpoint_sha
    out.mkdir()
    x, y, w, used = [], [], [], []
    for index, tensors in enumerate(tensor_samples(split["training"])):
        target, refs, truth = images(tensors)
        if not all(structure_score(target, truth[k])["passed"] for k in (0, 2)):
            continue
        area = neutral(target)
        take = np.zeros_like(area)
        take[::4, ::4] = True
        area &= take
        if not area.any():
            continue
        protection_a, protection_b = masks(target, [truth[0], truth[2]])
        switch = (protection_a | protection_b)[area]
        used.append(index)
        for ref, authored in zip(refs, truth, strict=True):
            baseline = finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
            x.append(features(target, ref, baseline, area))
            y.append((get_color(authored)-get_color(baseline))[area])
            w.append(switch)
    x, y, w = map(np.concatenate, (x, y, w))
    if not w.any():
        raise ValueError("No aligned switch pixels")
    coef = {}
    for arm, weight in (("uniform", np.ones(len(w))), ("switch", 1. + 3. * w)):
        coef[arm] = np.linalg.solve(x.T @ (weight[:, None] * x) + .01 * weight.sum() * np.eye(x.shape[1]),
                                   x.T @ (weight[:, None] * y))
    results = {}
    for index, tensors in enumerate(tensor_samples(split["development"])):
        target, refs, truth = images(tensors)
        protections = masks(target, [truth[0], truth[2]]) if index in cases else None
        source = [checked_rgb(archive / f"{index}-{k}.png", stable["predictions"][f"{index}-{k}.png"]) for k in range(3)]
        all_arms = {"base": source}
        for arm, fit in coef.items():
            all_arms[arm] = [grade(target, ref, prev, fit) for ref, prev in zip(refs, source, strict=True)]
            np.testing.assert_array_equal(grade(target, target.copy(), target, fit), target)
            for prev, result in zip(source, all_arms[arm], strict=True):
                np.testing.assert_array_equal(prev[~neutral(target)], result[~neutral(target)])
        expected = truth[2].astype(float) - truth[0]
        results[index] = {}
        for arm, frames in all_arms.items():
            response = frames[2].astype(float) - .5 * (frames[0].astype(float) + frames[1])
            projected = float(np.sum(response * expected) / np.square(expected).sum()) if np.any(expected) else None
            same = float(np.abs(frames[0].astype(float) - frames[1]).mean())
            distinctions = [float(np.abs(frames[2].astype(float) - v).mean()) for v in frames[:2]]
            errors = [float(np.abs(a.astype(float) - b).mean()) for a, b in zip(frames, truth, strict=True)]
            result = dict(whole_error=errors, whole_projection=projected, same_swap=same,
                          whole_pass=bool(np.abs(expected).mean() >= 1 and same < min(distinctions) and projected >= .5))
            if arm == "base":
                assert result["whole_pass"] == stable["cases"][index]["passed"]
            if protections:
                result["switch"] = measure(frames, [truth[0], truth[2]], protections)
            if arm != "base":
                result["outputs"] = [save(out / f"{index}-{arm}-{k}.png", v) for k, v in enumerate(frames)]
            results[index][arm] = result
    summary = {}
    for arm in ("base", *coef):
        records = [row[arm] for row in results.values()]
        summary[arm] = dict(whole_pass=sum(row["whole_pass"] for row in records),
                            switched_pass=sum(row.get("switch", {}).get("passed", False) for row in records),
                            max_whole_regression=max(np.max(np.array(row[arm]["whole_error"]) - row["base"]["whole_error"])
                                                     for row in results.values()))
    json_to(out / "results.json", dict(training_groups=used, rows=len(w), weighted_rows=int(w.sum()),
            coefficients={k: v.tolist() for k, v in coef.items()}, source_sha256=RENDER_SHA,
            split_sha256=SPLIT_SHA, cases=results, summary=summary))
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
