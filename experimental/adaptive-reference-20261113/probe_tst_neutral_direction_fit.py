"""Fit chroma direction and amplitude separately from ordinary two-image inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_tst_neutral_switch_fit import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, SpatialEngine,
    checked_rgb, features, finish_spatial, get_color, images, json_to, load, masks,
    measure, neutral, render, save, structure_score, tensor_samples,
)


def targets(target, truth, area):
    vector = (get_color(truth) - get_color(target))[area]
    has_edit = np.abs(vector).max(axis=1) > 1
    vector = np.where(has_edit[:, None], vector, 0)
    length = np.linalg.norm(vector, axis=1, keepdims=True)
    unit = np.divide(vector, length, out=np.zeros_like(vector), where=length > 0)
    return np.column_stack((vector, unit, length))


def fit(split, stable):
    engine = SpatialEngine("multiref-v1")
    if stable["checkpoint_sha256"] != engine.checkpoint_sha:
        raise ValueError("Unexpected 13 checkpoint")
    features_rows, labels, used = [], [], []
    for index, tensors in enumerate(tensor_samples(split["training"])):
        target, refs, truths = images(tensors)
        if not all(structure_score(target, truths[k])["passed"] for k in (0, 2)):
            continue
        area = neutral(target)
        grid = np.zeros_like(area)
        grid[::4, ::4] = True
        area &= grid
        if not area.any():
            continue
        used.append(index)
        for ref, truth in zip(refs, truths, strict=True):
            baseline = finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
            features_rows.append(features(target, ref, baseline, area))
            labels.append(targets(target, truth, area))
    x, y = np.concatenate(features_rows), np.concatenate(labels)
    if len(used) != 82 or len(x) != 114156:
        raise ValueError("Expected frozen training set and sampling")
    coefficients = np.linalg.solve(x.T @ x + .01 * len(x) * np.eye(x.shape[1]), x.T @ y)
    return coefficients, {"groups": used, "rows": len(x), "unchanged_rows": int((y[:, -1] == 0).sum())}


def predict(target, reference, baseline, coefficients):
    if np.array_equal(target, reference):
        return {"vector": target.copy(), "direction": target.copy()}, {}
    area = neutral(target)
    if not area.any():
        return {"vector": baseline.copy(), "direction": baseline.copy()}, {}
    estimated = features(target, reference, baseline, area) @ coefficients
    original = get_color(target)[area]
    upper = 2 * np.linalg.norm(get_color(baseline)[area] - original, axis=1)
    length = estimated[:, 4].clip(0, upper)
    light = (baseline @ LUMA)[area]
    outputs, limit = {}, {}
    for name, vector in (("vector", estimated[:, :2]), ("direction", estimated[:, 2:4])):
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        unit = np.divide(vector, norm, out=np.zeros_like(vector), where=norm > 1e-12)
        pixels, limited = render(original + unit * length[:, None], light)
        output = baseline.copy()
        output[area] = pixels
        outputs[name] = output
        limit[name] = limited
    return outputs, limit


def whole(frames, source, truths):
    expected = truths[2].astype(float) - truths[0]
    actual = frames[2].astype(float) - .5 * (frames[0].astype(float) + frames[1])
    projected = float(np.vdot(actual, expected) / np.vdot(expected, expected)) if np.any(expected) else None
    same = float(np.abs(frames[0].astype(float) - frames[1]).mean())
    different = min(float(np.abs(frames[2].astype(float) - reference).mean()) for reference in frames[:2])
    errors = [float(np.abs(frame.astype(float) - truth).mean()) for frame, truth in zip(frames, truths, strict=True)]
    baseline = [float(np.abs(frame.astype(float) - truth).mean()) for frame, truth in zip(source, truths, strict=True)]
    return {"mae": errors, "worst_regression": max(a - b for a, b in zip(errors, baseline, strict=True)),
            "projection": projected, "passed": bool(np.abs(expected).mean() >= 1 and same < different and projected >= .5)}


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    ready = {row["index"] for row in assay["cases"] if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    coefficients, training = fit(split, stable)
    out.mkdir()
    results = {}
    for index, tensors in enumerate(tensor_samples(split["development"])):
        target, refs, truths = images(tensors)
        source = [checked_rgb(archive / f"{index}-{i}.png", stable["predictions"][f"{index}-{i}.png"]) for i in range(3)]
        source_summary = whole(source, source, truths)
        if source_summary["passed"] != stable["cases"][index]["passed"]:
            raise ValueError(f"Changed baseline result {index}")
        area = neutral(target)
        selected = masks(target, (truths[0], truths[2])) if index in ready else None
        attempts = [predict(target, ref, frame, coefficients) for ref, frame in zip(refs, source, strict=True)]
        for frame in predict(target, target.copy(), target, coefficients)[0].values():
            np.testing.assert_array_equal(frame, target)
        rows = {"13": {"whole": source_summary}}
        if selected is not None:
            rows["13"]["local"] = measure(source, (truths[0], truths[2]), selected)
        for name in ("vector", "direction"):
            outputs = [value[0][name] for value in attempts]
            evidence = []
            for slot, (before, changed) in enumerate(zip(source, outputs, strict=True)):
                np.testing.assert_array_equal(changed[~area], before[~area])
                evidence.append(save(out / f"{index}-{slot}-{name}.png", changed))
            row = {"whole": whole(outputs, source, truths), "hashes": evidence,
                   "limited_fraction": [info.get(name, 0.) for _, info in attempts]}
            if selected is not None:
                row["local"] = measure(outputs, (truths[0], truths[2]), selected)
            rows[name] = row
        results[str(index)] = rows
    if len(results) != 24 or len(ready) != 5:
        raise ValueError("Unexpected evaluation cohort")
    summary = {}
    for name in ("13", "vector", "direction"):
        arms = [case[name] for case in results.values()]
        summary[name] = {"whole_pass": sum(arm["whole"]["passed"] for arm in arms),
                         "local_pass": sum(arm.get("local", {}).get("passed", False) for arm in arms),
                         "worst_regression": max(arm["whole"]["worst_regression"] for arm in arms)}
    candidate = summary["direction"]
    passed = candidate["local_pass"] >= 3 and candidate["whole_pass"] >= summary["13"]["whole_pass"] and candidate["worst_regression"] <= 2
    json_to(out / "results.json", {"split_sha256": SPLIT_SHA, "baseline_sha256": RENDER_SHA, "training": training,
                                   "coefficients": coefficients.tolist(), "cases": results, "summary": summary,
                                   "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
