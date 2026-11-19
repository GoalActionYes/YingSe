"""Fit the same two-image chroma head with and without donor-edit supervision."""
import argparse
from pathlib import Path

import numpy as np
import torch

from audit_tst_reference_equivalence import COHORT, HASHES, family
from probe_inretouch_registration import UnmeasurableRegistration
from probe_tst_neutral_direction_fit import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, SpatialEngine,
    checked_rgb, features, finish_spatial, get_color, images, json_to, load, masks,
    measure, neutral, render, save, structure_score, tensor_samples, whole,
)
from probe_tst_paired_donor_edit import donor_assets, edit_number, replay

ARMS = ("author", "donor")


def representative():
    parents = {}

    def root(name):
        while name in parents:
            name = parents[name]
        return name

    for a, b in load(COHORT, HASHES[1])["alias_pairs"]:
        first, second = sorted((root(a), root(b)))
        if first != second:
            parents[second] = first
    return root


def fit(split, frozen):
    root = representative()
    protected = {root(family(path, role)) for row in split["development"]
                 for role, path in [("content", row["content"]), ("gt", row["same_truth"]),
                                    ("gt", row["different_truth"]), *(("reference", p) for p in row["refs"])]}
    asset_map, read = donor_assets()
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != frozen["checkpoint_sha256"]:
        raise ValueError("Unexpected baseline weights")
    yy, xx = np.indices((256, 256))
    train = (xx // 16 + yy // 16) % 2 == 0
    design, assigned, slots, flat = [], [], [], []
    for index, row in enumerate(split["training"]):
        target, target_shape = read(row["content"])
        area = neutral(target)
        area &= (yy % 4 == 0) & (xx % 4 == 0)
        if not area.any():
            continue
        for slot, ref in enumerate(row["refs"]):
            author_path = row["different_truth" if slot == 2 else "same_truth"]
            family_id = family(ref, "reference")
            originals = asset_map[family_id]
            if len(originals) != 1 or edit_number(ref) != edit_number(author_path):
                continue
            if {root(family_id), root(family(row["content"], "content")),
                    root(family(author_path, "gt"))} & protected:
                raise ValueError("Development source entered fitting")
            reference, ref_shape = read(ref)
            original, donor_shape = read(next(iter(originals)))
            author, author_shape = read(author_path)
            if ref_shape != donor_shape or target_shape != author_shape:
                continue
            try:
                registered = (structure_score(original, reference)["passed"]
                              and structure_score(target, author)["passed"])
            except UnmeasurableRegistration:
                flat.append([index, slot])
                continue
            if not registered:
                continue
            teacher = replay(original, reference, target, train)
            blended = finish_spatial(target, author, .8)
            base = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
            source = get_color(target)[area]
            teacher_delta = get_color(teacher)[area] - source
            author_delta = get_color(blended)[area] - source
            design.append(features(target, reference, base, area))
            assigned.append(np.column_stack((author_delta, .5 * (author_delta + teacher_delta))))
            slots.append([index, slot])
    x, y = np.concatenate(design), np.concatenate(assigned)
    coef = np.linalg.solve(x.T @ x + .01 * len(x) * np.eye(x.shape[1]), x.T @ y)
    return coef, {"slots": slots, "rows": len(x), "groups": len({index for index, _ in slots}),
                  "unmeasurable_registration": flat}


def grade(target, reference, baseline, coef, area, arms=ARMS):
    if np.array_equal(target, reference):
        return {key: target.copy() for key in arms}
    outputs = {key: baseline.copy() for key in arms}
    if not area.any():
        return outputs
    delta = features(target, reference, baseline, area) @ coef
    original = get_color(target)[area]
    length_limit = 2 * np.linalg.norm(get_color(baseline)[area] - original, axis=1)
    light = (baseline @ LUMA)[area]
    for name, vectors in zip(arms, np.split(delta, len(arms), axis=1), strict=True):
        length = np.linalg.norm(vectors, axis=1)
        factor = np.divide(length_limit, length, out=np.ones_like(length), where=length > 0)
        outputs[name][area], _ = render(original + vectors * np.minimum(factor, 1)[:, None], light)
    return outputs


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    archive = BASE / "reference-equivalence-development-baseline-v1"
    frozen = load(archive / "results.json", RENDER_SHA)
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    ready = {row["index"] for row in assay["cases"] if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    if len(split["training"]) != 82 or len(split["development"]) != 24 or len(ready) != 5:
        raise ValueError("Published cohort changed")
    coef, training = fit(split, frozen)
    out.mkdir()
    results = {}
    for index, tensors in enumerate(tensor_samples(split["development"])):
        target, refs, truths = images(tensors)
        area = neutral(target)
        baseline = [checked_rgb(archive / f"{index}-{i}.png", frozen["predictions"][f"{index}-{i}.png"])
                    for i in range(3)]
        attempts = [grade(target, ref, frame, coef, area) for ref, frame in zip(refs, baseline, strict=True)]
        for name in ARMS:
            np.testing.assert_array_equal(grade(target, target, baseline[0], coef, area)[name], target)
            np.testing.assert_array_equal(
                grade(target, refs[0], baseline[0], coef, np.zeros_like(area))[name], baseline[0]
            )
        selected = masks(target, (truths[0], truths[2])) if index in ready else None
        rows = {}
        for name in ("13", *ARMS):
            frames = baseline if name == "13" else [arm[name] for arm in attempts]
            scores = {"whole": whole(frames, baseline, truths)}
            if selected is not None:
                scores["local"] = measure(frames, (truths[0], truths[2]), selected)
            if name == "13" and scores["whole"]["passed"] != frozen["cases"][index]["passed"]:
                raise ValueError("Baseline decision changed")
            if name != "13":
                scores["hashes"] = []
                for slot, (before, after) in enumerate(zip(baseline, frames, strict=True)):
                    np.testing.assert_array_equal(after[~area], before[~area])
                    scores["hashes"].append(save(out / f"{index}-{slot}-{name}.png", after))
            rows[name] = scores
        results[str(index)] = rows
    summary = {}
    for name in ("13", *ARMS):
        arms = [row[name] for row in results.values()]
        summary[name] = {"whole_pass": sum(row["whole"]["passed"] for row in arms),
                         "local_pass": sum(row.get("local", {}).get("passed", False) for row in arms),
                         "worst_regression": max(row["whole"]["worst_regression"] for row in arms)}
    candidate = summary["donor"]
    passed = (candidate["local_pass"] >= 3 and candidate["local_pass"] >= summary["author"]["local_pass"] + 2
              and candidate["whole_pass"] >= 7 and candidate["worst_regression"] <= 2)
    json_to(out / "results.json", {"training": training, "cases": results, "summary": summary,
                                  "coefficients": coef.tolist(), "passed": passed,
                                  "split_sha256": SPLIT_SHA, "baseline_sha256": RENDER_SHA})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
