"""Measure shared and complementary errors between two same-look references."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST, load
from probe_hdm_hdr_display import json_to
from probe_sparks_shared_chroma import LUMA, render
from probe_target_mirror_chroma import checked_rgb, panel, save
from probe_tst_authored_neutral_intent import RENDER_SHA, get_color
from probe_tst_neutral_switch_renders import masks, measure
from spatial_render import finish_spatial
from train_tst_reference_equivalence import BASE, SPLIT, SPLIT_SHA

FROZEN = "8b5a838705854dc2cca58db079ae5cc2aeb76dcbcd12c3b3aac4564a458e50f3"


def optimum(a, b, truth):
    delta = b - a
    roots = np.divide(truth - a, delta, out=np.zeros_like(delta), where=delta != 0).clip(0, 1)
    candidates = (a, b, *(a + delta * roots[:, i:i + 1] for i in range(delta.shape[1])))
    outcomes = np.stack(candidates, axis=1)
    absolute = np.abs(outcomes - truth[:, None]).mean(axis=2)
    choice = absolute.argmin(axis=1)
    return outcomes[np.arange(len(choice)), choice]


def fused(first, second, truth, selected, oracle):
    a, b, want = (get_color(frame)[selected] for frame in (first, second, truth))
    colors = optimum(a, b, want) if oracle else .5 * (a + b)
    base = .5 * ((first @ LUMA)[selected] + (second @ LUMA)[selected])
    encoded, clipped = render(colors, base)
    output = first.copy()
    output[selected] = encoded
    return output, {"floating_chroma_mae": float(np.abs(colors - want).mean()), "limited_fraction": clipped}


def qualified(scores, whole_delta):
    return (scores["preserve_mae"] <= 1.5 and .75 <= scores["style_projection"] <= 1.25
            and scores["style_delta_mae"] <= 3 and whole_delta <= 2)


def frozen_cases():
    archived = load(BASE.parent / "tst-neutral-switch-renders-v1/results.json", FROZEN)
    assignments = load(SPLIT / "private-assignment.json", SPLIT_SHA)["development"]
    entries = {item["original_path"]: item for item in load(MANIFEST, HASHES[2])["files"]}
    directory = BASE / "reference-equivalence-development-baseline-v1"
    predictions = load(directory / "results.json", RENDER_SHA)["predictions"]

    def rgb(key):
        row = entries[key]
        value = checked_rgb(COHORT.parent / row["prepared_path"], row["sha256"])
        return np.asarray(Image.fromarray(value).resize((256, 256), Image.Resampling.BILINEAR))

    for key, archive in archived["cases"].items():
        assignment = assignments[int(key)]
        original = rgb(assignment["content"])
        first_truth, other_truth = (finish_spatial(original, rgb(assignment[field]), .8)
                                    for field in ("same_truth", "different_truth"))
        protection = masks(original, (first_truth, other_truth))
        if [int(area.sum()) for area in protection] != archive["protect_counts"]:
            raise ValueError(f"Frozen regions changed: {key}")
        names = [f"{key}-{slot}.png" for slot in range(3)]
        first, second, other = (checked_rgb(directory / name, predictions[name]) for name in names)
        baseline_scores = measure((first, second, other), (first_truth, other_truth), protection)
        if baseline_scores != archive["scores"]["13"]:
            raise ValueError(f"Frozen scores changed: {key}")
        yield key, original, (first_truth, other_truth), (first, second, other), protection, baseline_scores


def run(out):
    out.mkdir()
    records = {}
    for key, _, (first_truth, other_truth), (first, second, other), protection, baseline_scores in frozen_cases():
        selected = protection[0] | protection[1]
        e0, e1 = (get_color(frame)[selected] - get_color(first_truth)[selected] for frame in (first, second))
        shared = ((np.sign(e0) == np.sign(e1)) & (np.minimum(np.abs(e0), np.abs(e1)) > 1))
        report, board = {}, {"13 first": first, "13 second": second, "author": first_truth}
        ordinary = .5 * sum(np.abs(frame.astype(np.int16) - first_truth).mean() for frame in (first, second))
        for arm, oracle in (("mean", False), ("oracle", True)):
            output, extra = fused(first, second, first_truth, selected, oracle)
            scores = measure((output, output, other), (first_truth, other_truth), protection)
            scores.pop("passed")
            scores.pop("same_style_swap_mae")
            added = float(np.abs(output.astype(np.int16) - first_truth).mean() - ordinary)
            report[arm] = {**extra, "scores": scores, "whole_mae_delta": added,
                           "passed": qualified(scores, added), "sha256": save(out / f"{key}-{arm}.png", output)}
            board[arm] = output
        records[key] = {"reference_13": baseline_scores, "shared_error_fraction": float(shared.mean()),
                        "arms": report, "image_sha256": panel(out / f"{key}-comparison.png", {"Same look": board})}
    if len(records) != 5:
        raise ValueError("Expected frozen five cases")
    counts = {arm: sum(row["arms"][arm]["passed"] for row in records.values()) for arm in ("mean", "oracle")}
    json_to(out / "results.json", {"parent_sha256": FROZEN, "cases": records, "passed_cases": counts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
