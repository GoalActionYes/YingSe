"""Render complete look transforms inferred from a training-only paired-photo library."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_tst_reference_equivalence import family
from probe_aligned_cubic_pilot import cubic_mapping, learn
from probe_inretouch_registration import UnmeasurableRegistration
from probe_tst_neutral_direction_fit import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, checked_rgb, get_color, images,
    json_to, load, masks, measure, save, structure_score, tensor_samples, whole,
)
from probe_tst_donor_supervision import representative
from probe_tst_paired_donor_edit import donor_assets, edit_number
from reference_color_app.lut import apply_mapping


def describe(rgb):
    signals = np.column_stack((rgb.reshape(-1, 3), (rgb @ LUMA).ravel(), get_color(rgb).reshape(-1, 2)))
    return np.quantile(signals, [.1, .3, .5, .7, .9], axis=0).ravel()


def build_library(split):
    _, read = donor_assets()
    root = representative()
    protected = {root(family(path, role)) for row in split["development"]
                 for role, path in [("content", row["content"]), ("gt", row["same_truth"]),
                                    ("gt", row["different_truth"]), *(("reference", p) for p in row["refs"])]}
    yy, xx = np.indices((256, 256))
    train = (xx // 16 + yy // 16) % 2 == 0
    entries = defaultdict(dict)
    audit = Counter()
    for row in split["training"]:
        original, orig_shape = read(row["content"])
        source = root(family(row["content"], "content"))
        for name in ("same_truth", "different_truth"):
            path = row[name]
            label = edit_number(path)
            if source in protected or root(family(path, "gt")) in protected:
                raise ValueError("Development family in the fitting library")
            if source in entries[label]:
                audit["duplicate"] += 1
                continue
            edited, shape = read(path)
            try:
                aligned = shape == orig_shape and structure_score(original, edited)["passed"]
            except UnmeasurableRegistration:
                aligned = False
            if not aligned:
                audit["unaligned"] += 1
                continue
            weights = learn(original, edited, train.ravel())
            if weights is None:
                audit["rank_failed"] += 1
                continue
            preview = apply_mapping(original, cubic_mapping(weights), 1.)
            error = np.abs(preview[~train].astype(np.int16) - edited[~train])
            if error.mean() > 1.5 or np.quantile(error, .95) > 4:
                audit["holdout_failed"] += 1
                continue
            entries[label][source] = (describe(edited), weights)
            audit["qualified"] += 1
    kept = {label: list(items.values()) for label, items in sorted(entries.items()) if len(items) >= 2}
    if len(kept) < 2:
        return None, dict(audit) | {"styles": {k: len(v) for k, v in kept.items()}}
    examples = {label: np.stack([d for d, _ in items]) for label, items in kept.items()}
    scale = np.maximum(np.concatenate(list(examples.values())).std(axis=0), 1.)
    examples = {label: x / scale for label, x in examples.items()}
    within = []
    for x in examples.values():
        distances = np.linalg.norm(x[:, None] - x, axis=2)
        np.fill_diagonal(distances, np.inf)
        within.extend(distances.min(axis=1))
    radius = float(np.quantile(within, .9))
    maps = {label: cubic_mapping(np.median(np.stack([weights for _, weights in items]), axis=0))
            for label, items in kept.items()}
    return (examples, scale, radius, maps), dict(audit) | {"styles": {k: len(v) for k, v in kept.items()},
                                                          "radius": radius}


def predict(target, reference, baseline, library):
    if np.array_equal(target, reference):
        return target.copy(), {"route": "same_image"}
    if library is None:
        return baseline.copy(), {"route": "fallback_small_library"}
    examples, scale, radius, maps = library
    query = describe(reference) / scale
    (first, label), (second, _) = sorted((float(np.linalg.norm(x - query, axis=1).min()), item)
                                         for item, x in examples.items())[:2]
    accepted = second > 0 and first <= radius and first <= .8 * second
    picture = apply_mapping(target, maps[label], .8) if accepted else baseline.copy()
    return picture, {"route": "library" if accepted else "fallback_distance", "closest": label,
                     "distance": first, "second_distance": second}


def run(out):
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    assay = load(ROOT / "docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json", ASSAY)
    ready = {row["index"] for row in assay["cases"] if row["aligned"] and row["one_state_protects_pixels"] >= 256}
    if len(split["training"]) != 82 or len(split["development"]) != 24 or len(ready) != 5:
        raise ValueError("Frozen development cohort changed")
    library, audit = build_library(split)
    supported = set(audit["styles"])
    counts = Counter()
    out.mkdir()
    results = {}
    for index, (row, tensors) in enumerate(zip(split["development"], tensor_samples(split["development"]), strict=True)):
        target, refs, truths = images(tensors)
        frozen = [checked_rgb(archive / f"{index}-{slot}.png", stable["predictions"][f"{index}-{slot}.png"])
                  for slot in range(3)]
        np.testing.assert_array_equal(predict(target, target.copy(), frozen[0], library)[0], target)
        outputs, receipts = [], []
        for slot, (reference, baseline, path) in enumerate(zip(refs, frozen, row["refs"], strict=True)):
            frame, record = predict(target, reference, baseline, library)
            label = edit_number(path)
            used = record["route"] == "library"
            known = label in supported
            counts["known" if known else "unknown"] += 1
            if used:
                counts["accepted"] += 1
                counts["known_correct" if known and record["closest"] == label else "accepted_wrong"] += 1
                counts["unknown_accepted"] += int(not known)
            elif record["route"] != "same_image":
                np.testing.assert_array_equal(frame, baseline)
            outputs.append(frame)
            receipts.append(record | {"actual": label, "sha256": save(out / f"{index}-{slot}.png", frame)})
        case = {"13": {"whole": whole(frozen, frozen, truths)},
                "library": {"whole": whole(outputs, frozen, truths)}, "receipts": receipts}
        if case["13"]["whole"]["passed"] != stable["cases"][index]["passed"]:
            raise ValueError("Frozen baseline decision changed")
        if index in ready:
            region = masks(target, (truths[0], truths[2]))
            for name, frames in (("13", frozen), ("library", outputs)):
                case[name]["local"] = measure(frames, (truths[0], truths[2]), region)
        results[str(index)] = case
    summary = {}
    for name in ("13", "library"):
        arms = [row[name] for row in results.values()]
        summary[name] = {"whole_pass": sum(row["whole"]["passed"] for row in arms),
                         "local_pass": sum(row.get("local", {}).get("passed", False) for row in arms),
                         "worst_regression": max(row["whole"]["worst_regression"] for row in arms)}
    candidate = summary["library"]
    passed = candidate["local_pass"] >= 3 and candidate["whole_pass"] >= summary["13"]["whole_pass"] and candidate["worst_regression"] <= 2
    json_to(out / "results.json", {"split_sha256": SPLIT_SHA, "baseline_report_sha256": RENDER_SHA,
                                  "training": audit, "recognition": dict(counts), "summary": summary,
                                  "cases": results, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
