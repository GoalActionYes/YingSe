"""Compare real capture exposure diversity with repeated nominal training data."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, describe, digest, feature, fit, grade, headroom, mask_for, pair, panel, read, scores, target_regions,
)
from probe_bg_real_paired_contrast import write_outputs
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import STATES, save_report, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, read_ledger

ARMS = ("nominal", "real_exposures")


def base_rows(fitting):
    targets = [describe(row["C1"]) for row in fitting]
    features, labels = [], []
    for index, group in enumerate(fitting):
        target = targets[(index + 1) % len(targets)]
        for operation, photo in product(EDITS.values(), group.values()):
            features.append(feature(target, describe(grade(photo, operation))))
            labels.append(operation)
    return features, labels


def exposure_rows(folder):
    groups = read_ledger(folder, "inputs.json", INPUT_SHA)["groups"]
    photos = {row["group"]: {state: headroom(verified_png(folder / str(row["group"]) / f"{state}.png",
                                                        row["frames"][state])) for state in STATES}
              for row in groups}
    targets = {group: describe(pictures["nominal"]) for group, pictures in photos.items()}
    rows, labels = {arm: [] for arm in ARMS}, []
    for group in groups:
        references = photos[group["group"]]
        target = targets[group["target_group"]]
        for operation in EDITS.values():
            nominal = feature(target, describe(grade(references["nominal"], operation)))
            for state in STATES:
                rows["nominal"].append(nominal)
                actual = nominal if state == "nominal" else feature(target, describe(grade(references[state], operation)))
                rows["real_exposures"].append(actual)
                labels.append(operation)
    return rows, labels


def adjust(model, target, reference, canvas):
    center, scale, intercept, weights = model
    estimates = (feature(target, reference) - center) / scale @ weights + intercept
    bounded = np.clip(estimates, [-1, 0], [1, 1])
    return grade(canvas, bounded, .8), bounded.tolist()


def run(output):
    fitting, held, known = fixed_sets()
    features, labels = base_rows(fitting)
    extra, extra_labels = exposure_rows(BASE / "dast-exposure-prepared-v1")
    if len(labels) != 96 or len(extra_labels) != 120:
        raise ValueError("Unexpected frozen fitting budget")
    models = {arm: fit(features + rows, labels + extra_labels) for arm, rows in extra.items()}
    output.mkdir()
    cases = {}
    for example in held.values():
        key = example["target_pair"]
        target_info = known[key]
        source = read(target_info["input_file"])
        canvas = headroom(source)
        target = describe(canvas)
        refs = pair(BASE / "bg-real-pixel-inputs", example["input"])
        observations = {role: {style: describe(grade(rgb, edit)) for style, edit in EDITS.items()}
                        for role, rgb in refs.items()}
        results, parameters = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
        for arm, model in models.items():
            for role, styles in observations.items():
                results[arm][role], parameters[arm][role] = {}, {}
                for style, stats in styles.items():
                    results[arm][role][style], parameters[arm][role][style] = adjust(model, target, stats, canvas)

        human = mask_for(target_info, key.split("-")[0])
        neutral = target_regions(source, human)["neutral_proxy"] & human
        expected = {style: grade(canvas, edit, .8) for style, edit in EDITS.items()}
        verdicts = {arm: scores(photos, expected, human, neutral) for arm, photos in results.items()}
        regression = max(verdicts["real_exposures"]["styles"][style]["response"][role]["target_mae"]
                         - verdicts["nominal"]["styles"][style]["response"][role]["target_mae"]
                         for style, role in product(EDITS, refs))
        dest = output / key
        dest.mkdir()
        hashes = {arm: write_outputs(dest / arm, data) for arm, data in results.items()}
        sheet = {"prescribed edit": expected}
        sheet.update({f"{arm} / {role}": photos for arm, data in results.items() for role, photos in data.items()})
        panel_file = dest / "comparison.png"
        panel(panel_file, sheet)
        cases[key] = {**verdicts, "estimated": parameters, "output_sha256": hashes,
                      "worst_regression": regression, "panel_sha256": digest(panel_file)}

    count = {arm: sum(row[arm]["passed"] for row in cases.values()) for arm in ARMS}
    worst = max(row["worst_regression"] for row in cases.values())
    save_report(output / "results.json", {"background_training_rows": len(labels), "exposure_training_rows": len(extra_labels),
                "capture_input_sha256": INPUT_SHA, "totals": count, "cases": cases, "worst_regression": worst,
                "passed": count["real_exposures"] >= 3 and count["real_exposures"] > count["nominal"] and worst <= 2.})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
