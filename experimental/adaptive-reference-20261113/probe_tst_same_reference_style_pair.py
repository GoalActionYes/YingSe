"""Hold the donor photo fixed while swapping two author-indexed edit numbers."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit_tst_reference_equivalence import HASHES, INDEX, MANIFEST, family
from probe_inretouch_registration import UnmeasurableRegistration
from probe_tst_donor_supervision import grade
from probe_tst_neutral_direction_fit import (
    BASE, LUMA, RENDER_SHA, SPLIT, SPLIT_SHA, SpatialEngine, checked_rgb, finish_spatial,
    get_color, json_to, load, neutral, save, structure_score,
)
from probe_tst_paired_donor_edit import donor_assets, edit_number

DONOR_SHA = "59d9b5c0eca38d63177a5c2daa7d1f479c9023316e04dfdcc05138159fc5cd1b"


def aligned(first, second):
    original, first_shape = first
    revised, second_shape = second
    if first_shape != second_shape:
        return False
    try:
        return structure_score(original, revised)["passed"]
    except UnmeasurableRegistration:
        return False


def measure(frames, authored, region):
    left, right = (image.astype(float) for image in frames)
    low, high = (image.astype(float) for image in authored)
    result = {}
    for label, convert in (("luma", lambda x: x @ LUMA), ("chroma", get_color)):
        actual = (convert(right) - convert(left))[region]
        expected = (convert(high) - convert(low))[region]
        norm = float(np.abs(expected).mean())
        if norm < 1:
            result[label] = {"author_amount": norm, "scored": False}
            continue
        projection = float(np.vdot(actual, expected) / np.vdot(expected, expected))
        error = float(np.abs(actual - expected).mean())
        result[label] = {"author_amount": norm, "scored": True, "projection": projection,
                         "mae": error, "passed": .75 <= projection <= 1.25 and error <= 3}
    return result


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    split = load(SPLIT / "private-assignment.json", SPLIT_SHA)
    ready = {row["original_path"] for row in load(MANIFEST, HASHES[2])["files"]}
    catalog = defaultdict(set)
    for row in load(INDEX, HASHES[0]):
        for role in ("reference", "gt"):
            path = row[role]
            catalog[family(path, role), edit_number(path)].add(path)
    _, read = donor_assets()
    engine = SpatialEngine("multiref-v1")
    archive = BASE / "reference-equivalence-development-baseline-v1"
    stable = load(archive / "results.json", RENDER_SHA)
    trained = load(BASE.parent / "tst-donor-supervision-v1/results.json", DONOR_SHA)
    if engine.checkpoint_sha != stable["checkpoint_sha256"]:
        raise ValueError("Candidate weights changed")
    coef = np.array(trained["coefficients"])
    out.mkdir()
    cases = {}
    eligible = 0
    for index, row in enumerate(split["development"]):
        original = read(row["content"])
        target = original[0]
        authored = [read(row[key]) for key in ("same_truth", "different_truth")]
        edit_numbers = [edit_number(row[key]) for key in ("same_truth", "different_truth")]
        regions = {"whole": np.ones(target.shape[:2], bool), "neutral": neutral(target)}
        eligible_regions = {key: region for key, region in regions.items() if region.sum() >= 256}
        expected = [finish_spatial(target, img, .8) for img, _ in authored]
        target_aligned = all(aligned(original, truth) for truth in authored)
        old = [checked_rgb(archive / f"{index}-{i}.png", stable["predictions"][f"{index}-{i}.png"])
               for i in (0, 2)]
        slots = {}
        for slot, ref in enumerate(row["refs"]):
            group = family(ref, "reference")
            names = [catalog[group, number] for number in edit_numbers]
            complete = all(len(paths) == 1 and paths <= ready for paths in names)
            slot_info = {"index_matches": [len(paths) for paths in names], "available": complete,
                         "target_aligned": target_aligned}
            slots[str(slot)] = slot_info
            if not complete:
                continue
            eligible += 1
            first, second = [read(next(iter(paths))) for paths in names]
            slot_info["reference_aligned"] = aligned(first, second)
            if not (slot_info["reference_aligned"] and target_aligned):
                continue
            if np.abs(first[0].astype(float) - second[0]).mean() < 1:
                slot_info["reference_style_distinct"] = False
                continue
            slot_info["reference_style_distinct"] = True
            rated = {"13": [], "donor": []}
            for guide, _ in (first, second):
                baseline = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                prediction = grade(target, guide, baseline, coef, regions["neutral"])["donor"]
                rated["13"].append(baseline)
                rated["donor"].append(prediction)
            details = {}
            for name, frames in rated.items():
                measures = {key: measure(frames, expected, region) for key, region in eligible_regions.items()}
                judged = [item for scores in measures.values() for item in scores.values() if item["scored"]]
                excess = max(float(np.abs(frame.astype(float) - truth).mean()
                                   - np.abs(baseline.astype(float) - truth).mean())
                             for frame, truth, baseline in zip(frames, expected, rated["13"], strict=True))
                details[name] = {"areas": measures, "max_added_mae": excess,
                                 "hashes": [save(out / f"{index}-{slot}-{name}-{i}.png", frame)
                                            for i, frame in enumerate(frames)],
                                 "scored": bool(judged), "passed": bool(judged) and all(x["passed"] for x in judged) and excess <= 2}
            slot_info["arms"] = details
        cases[str(index)] = {"neutral_pixels": int(regions["neutral"].sum()), "slots": slots,
                             "original_context": {key: measure(old, expected, region)
                                                  for key, region in eligible_regions.items()} if target_aligned else {}}
    if len(cases) != 24 or eligible != 10:
        raise ValueError("Frozen cohort changed")
    completed = [[slot["arms"] for slot in row["slots"].values()
                  if "arms" in slot and slot["arms"]["13"]["scored"]] for row in cases.values()]
    scored = [group for group in completed if group]
    summary = {name: {"groups_pass": sum(all(slot[name]["passed"] for slot in group) for group in scored),
                      "max_added_mae": max((slot[name]["max_added_mae"] for group in scored for slot in group), default=None)}
               for name in ("13", "donor")}
    better = (len(scored) >= 5 and summary["donor"]["groups_pass"] >= summary["13"]["groups_pass"] + 2
              and summary["donor"]["max_added_mae"] <= 2)
    json_to(out / "results.json", {"split_sha256": SPLIT_SHA, "donor_sha256": DONOR_SHA, "eligible_slots": eligible,
                                  "scored_groups": len(scored), "scored_slots": sum(map(len, scored)),
                                  "summary": summary, "cases": cases, "passed": better})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
