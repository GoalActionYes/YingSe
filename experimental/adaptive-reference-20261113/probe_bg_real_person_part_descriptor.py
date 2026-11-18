"""Test automatic person-part observations in the fixed two-image edit fit."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, ROOT, describe, digest, feature, fit, grade, headroom, mask_for,
    pair, panel, read, scores, target_regions,
)
from probe_bg_real_paired_contrast import write_outputs
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import save_report
from segment import SHA256, Segmenter

ARMS = ("global", "parts")
PARTS = (3, 2, 4, 1)
LUMA = np.array([.2126, .7152, .0722])
BASELINE_SHA = "cbd75c0361b4a52068e9527dfb5a886c919843daf320434113dba2e6951de47b"


def summarize(pixels):
    light = pixels @ LUMA
    chroma = pixels[:, (0, 2)] - light[:, None]
    q25, q75 = np.percentile(light, (25, 75))
    return (*np.median(chroma, axis=0), q75 - q25)


def describe_both(rgb, locator):
    whole = describe(rgb)
    probabilities, _ = locator.predict(rgb)
    fallback = summarize(rgb.reshape(-1, 3))
    features, support = [], []
    for index in PARTS:
        eligible = probabilities[..., index] >= .7
        count = int(eligible.sum())
        present = count >= 256
        color_and_light = summarize(rgb[eligible]) if present else fallback
        features.extend((*color_and_light, float(present)))
        support.append(count)
    return {"global": whole, "parts": np.concatenate((whole, features))}, support


def fit_models(source, locator):
    targets = [describe_both(refs["C1"], locator)[0] for refs in source]
    x, y = {arm: [] for arm in ARMS}, []
    support = []
    for group, refs in enumerate(source):
        target = targets[(group + 1) % len(targets)]
        for operation, role in product(EDITS.values(), refs):
            observations, counts = describe_both(grade(refs[role], operation), locator)
            for arm in ARMS:
                x[arm].append(feature(target[arm], observations[arm]))
            y.append(operation)
            support.append(counts)
    if len(y) != 96:
        raise ValueError("Frozen fitting budget changed")
    return {arm: fit(rows, y) for arm, rows in x.items()}, np.sum(np.array(support) >= 256, axis=0).tolist()


def estimate(model, target, reference):
    center, scale, intercept, weights = model
    prediction = (feature(target, reference) - center) / scale @ weights + intercept
    return np.clip(prediction, [-1, 0], [1, 1])


def run(output, locator):
    fitting, held, saved = fixed_sets()
    previous = ROOT / "docs/research/2026-11-19-bg-real-crossed-grade-training-evidence.json"
    if digest(previous) != BASELINE_SHA:
        raise ValueError("Frozen global-arm evidence changed")
    prior = json.loads(previous.read_bytes())["cases"]
    models, training_support = fit_models(fitting, locator)
    output.mkdir()
    records = {}
    for case in held.values():
        key = case["target_pair"]
        entry = saved[key]
        original = read(entry["input_file"])
        canvas = headroom(original)
        target, target_support = describe_both(canvas, locator)
        refs = pair(BASE / "bg-real-pixel-inputs", case["input"])
        path = BASE / "bg-real-crossed-grade-training" / f"{key}.png"
        if digest(path) != prior[key]["panel_sha256"]:
            raise ValueError("Prior global fit panel changed")
        with Image.open(path) as image:
            baseline = np.asarray(image.convert("RGB"))

        renders = {arm: {} for arm in ARMS}
        estimated = {arm: {} for arm in ARMS}
        ref_support = {}
        for row, (role, rgb) in enumerate(refs.items()):
            ref_support[role] = {}
            for arm in ARMS:
                renders[arm][role], estimated[arm][role] = {}, {}
            for col, (style, operation) in enumerate(EDITS.items()):
                observation, ref_support[role][style] = describe_both(grade(rgb, operation), locator)
                for arm in ARMS:
                    coefficients = estimate(models[arm], target[arm], observation[arm])
                    adjusted = grade(canvas, coefficients, .8)
                    renders[arm][role][style] = adjusted
                    estimated[arm][role][style] = coefficients.tolist()
                    if arm == "global":
                        x, y = 8 + col * 264, 26 + (3 + row) * 280
                        np.testing.assert_array_equal(adjusted, baseline[y:y + 256, x:x + 256])

        human = mask_for(entry, key.split("-")[0])
        neutral = target_regions(original, human)["neutral_proxy"] & human
        truth = {style: grade(canvas, op, .8) for style, op in EDITS.items()}
        dest = output / key
        dest.mkdir()
        hashes = {arm: write_outputs(dest / arm, renders[arm]) for arm in ARMS}
        sheet = {"prescribed edit": truth}
        sheet.update({f"{arm} / {role}": frames for arm, data in renders.items() for role, frames in data.items()})
        panel_path = dest / "comparison.png"
        panel(panel_path, sheet)
        evaluation = {arm: scores(renders[arm], truth, human, neutral) for arm in ARMS}
        worst = max(evaluation["parts"]["styles"][style]["response"][role]["target_mae"]
                    - evaluation["global"]["styles"][style]["response"][role]["target_mae"]
                    for style, role in product(EDITS, refs))
        records[key] = {**evaluation, "estimated": estimated, "output_sha256": hashes,
                        "target_part_pixels": target_support, "reference_part_pixels": ref_support,
                        "panel_sha256": digest(panel_path), "worst_regression": worst}

    totals = {arm: sum(row[arm]["passed"] for row in records.values()) for arm in ARMS}
    worst = max(row["worst_regression"] for row in records.values())
    save_report(output / "results.json", {"cases": records, "totals": totals, "locator_sha256": SHA256,
                "training_qualified_by_part": training_support, "worst_regression": worst,
                "passed": totals["parts"] >= 3 and worst <= 2.})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("locator_dependencies", type=Path)
    args = cli.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    locator = Segmenter()
    try:
        run(args.output, locator)
    finally:
        locator.close()
