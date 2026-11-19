"""Test the color support of a selective reference without a subject detector."""

import argparse
from pathlib import Path

import numpy as np

from probe_arri_film_b_dose import verified
from probe_automatic_selective_chroma import PARENT_SHA, controls, render, save
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_psbench_background_desaturation import CHECKPOINT, TASKS, load, luminance_chroma, panel, regions, score


GUARDED_SHA = "5941f9a27fb91af45f8ec2b043ed86a427392fa85869c01cc8419924ed4a76a6"
BINS = 24


def angles(rgb):
    red, green, blue = np.moveaxis(rgb.astype(float), -1, 0)
    return np.mod(np.arctan2(np.sqrt(3) * (green - blue), 2 * red - green - blue), 2 * np.pi) * BINS / (2 * np.pi)


def weight(position, selected):
    centers = position - .5
    start = np.floor(centers).astype(int)
    fraction = centers - start
    return selected[start % BINS] * (1 - fraction) + selected[(start + 1) % BINS] * fraction


def hue_support(reference):
    _, color = luminance_chroma(reference)
    magnitude = np.linalg.norm(color, axis=2)
    vivid = magnitude >= 12
    gray = float((magnitude <= 3).mean())
    samples = int(vivid.sum())
    counts = np.bincount(angles(reference)[vivid].astype(int) % BINS, minlength=BINS)
    selected = counts >= max(32, .05 * samples)
    valid = gray >= .80 and samples >= max(512, .005 * magnitude.size) and bool(selected.any())
    return selected, {"near_gray_fraction": gray, "vivid_pixels": samples, "selected_bins": np.flatnonzero(selected).tolist(),
                      "applied": valid}


def candidate(target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"applied": False, "identity": True}, None
    selected, metadata = hue_support(reference)
    if not metadata["applied"]:
        return baseline.copy(), metadata, None
    retention = weight(angles(target), selected)
    result, delivery = render(target, baseline, retention)
    return result, metadata | delivery, retention


def self_check():
    select = np.zeros(BINS, bool)
    select[0] = True
    np.testing.assert_array_equal(weight(np.array([-.25, .25, BINS - .25, BINS + .25]), select),
                                  [.25, .75, .25, .75])
    rgb = np.full((40, 40, 3), 46, np.uint8)
    assert not hue_support(rgb)[1]["applied"]
    rgb[0, 0] = [250, 35, 20]
    _, limit = render(rgb, np.full_like(rgb, 2), np.ones(rgb.shape[:2]))
    assert limit["limited_fraction"] > 0 and limit["peak_luma_error"] <= .5


def run(base, out):
    self_check()
    upstream = base / "psbench-background-desaturation-v1"
    earlier = verified_json(upstream / "results.json", PARENT_SHA)
    preceding = base / "automatic-selective-chroma-v2"
    verified_controls = verified_json(preceding / "results.json", GUARDED_SHA)["negative"]
    sources = base / "psbench-background-desaturation-assets-v1"
    originals = {task: load(sources, f"{task}_start.jpg", earlier["input_files"][f"{task}_start.jpg"])[0]
                 for task in TASKS}
    out.mkdir()
    cases = {}
    for task, donor in (TASKS, TASKS[::-1]):
        donor_name = f"{donor}_end.jpg"
        reference = load(sources, donor_name, earlier["input_files"][donor_name])[0]
        target = originals[task]
        renders, diagnostics, weights = {}, {}, {}
        for role, guide in (("off", originals[donor]), ("on", reference)):
            baseline = verified(upstream / f"{task}-{role}.png", earlier["results"][task]["hashes"][role])
            renders[role], audit, weights[role] = candidate(target, guide, baseline)
            diagnostics[role] = audit | {"sha256": save(out / f"{task}-{role}.png", renders[role])}
        authored = f"{task}_end.jpg"
        truth = load(sources, authored, earlier["input_files"][authored])[0]
        masks = regions(target, truth)
        measured = score(target, truth, renders["off"], renders["on"], masks)
        for role, retention in weights.items():
            if retention is not None:
                diagnostics[role]["mean_retention_by_region"] = {region: float(retention[area].mean())
                                                                 for region, area in masks.items()}
        old = earlier["results"][task]["regions"]
        error = {region: item["absolute_chroma_errors"]["on"] for region, item in measured["regions"].items()}
        absolute_gate = (error["desaturated"] < old["desaturated"]["absolute_chroma_errors"]["on"]
                         and error["retained"] <= old["retained"]["absolute_chroma_errors"]["on"] + 1.5)
        cases[task] = measured | {"audit": diagnostics, "absolute_gate": absolute_gate,
                                 "panel_sha256": panel(out / f"{task}-comparison.png",
                                                       {"Target": target, "Edited reference": reference,
                                                        "Author output": truth, "13 (edited reference)": baseline,
                                                        "Hue candidate": renders["on"]}, masks)}

    negatives = {}
    for index, (name, reference) in enumerate(controls(base, {task: (rgb, None) for task, rgb in originals.items()})):
        target = originals[TASKS[index % 2]]
        expected_sha = verified_controls[name]["baseline_sha256"]
        baseline = verified(preceding / f"{name}-13.png", expected_sha)
        output, meta, _ = candidate(target, reference, baseline)
        good = not meta["applied"] and np.array_equal(output, baseline)
        negatives[name] = meta | {"passed": good, "sha256": save(out / f"{name}.png", output)}
    for task, target in originals.items():
        output, meta, _ = candidate(target, target.copy(), np.zeros_like(target))
        negatives[f"identity-{task}"] = meta | {
            "passed": np.array_equal(target, output), "sha256": save(out / f"identity-{task}.png", output)
        }
    if len(negatives) != 10:
        raise ValueError("Expected eight unrelated references and two same-image controls")
    passed = all(row["passed"] and row["absolute_gate"] for row in cases.values())
    passed &= all(row["passed"] for row in negatives.values())
    json_to(out / "results.json", {"psbench_sha256": PARENT_SHA, "previous_candidate_sha256": GUARDED_SHA,
                                   "baseline": CHECKPOINT, "cases": cases, "negative": negatives, "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
