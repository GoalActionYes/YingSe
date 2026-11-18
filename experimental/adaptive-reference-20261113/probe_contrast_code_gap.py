"""Test whether a single-reference contrast quantization cue survives export."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from probe_bg_real_crossed_grade_training import BASE, EDITS, grade, pair
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import STATES, digest, save_report, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, read_ledger

VARIANTS = {"lossless": (None, False), "jpeg95": (95, False), "jpeg85": (85, False), "resize_jpeg95": (95, True)}
PS_SHA = "9453038594bbd83b76fe51b2510b200608d123307e898b3fba65f33ef81e07ab"


def gap_score(rgb):
    observations = []
    pixels = rgb.reshape(-1, 3)
    min_count = len(pixels) * .0005
    for channel in pixels.T:
        hist = np.bincount(channel, minlength=256)
        supported = (hist[7:247] >= min_count) & (hist[9:249] >= min_count)
        observations.append(np.count_nonzero(supported & (hist[8:248] == 0)) / max(supported.sum(), 1))
    return float(np.mean(observations))


def export(rgb, quality, resized):
    if quality is None:
        return rgb
    encoded = BytesIO()
    image = Image.fromarray(rgb)
    if resized:
        image = image.resize(tuple(round(n * .8) for n in image.size), Image.Resampling.BILINEAR)
    image.save(encoded, format="JPEG", quality=quality, subsampling=2)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return np.asarray(decoded.convert("RGB"))


def measure(rgb):
    return {name: gap_score(export(rgb, *options)) for name, options in VARIANTS.items()}


def dast_negatives(source):
    inputs = read_ledger(source, "inputs.json", INPUT_SHA)
    rows = []
    for case in inputs["groups"]:
        identifier = case["group"]
        for state in STATES:
            photo = verified_png(source / str(identifier) / f"{state}.png", case["frames"][state])
            rows.append({"group": identifier, "exposure": state, "scores": measure(photo)})
    return rows


def authored_examples(folder):
    evidence = read_ledger(folder / "render-v2", "results.json", PS_SHA)
    rows = {}
    for record in evidence["source_checks"]:
        task = record["task"]
        rows[task] = {}
        for side, expected in zip(("start", "end"), record["files"], strict=True):
            source = folder / f"{task}_{side}.jpg"
            if digest(source) != expected["sha256"]:
                raise ValueError(f"Frozen instance changed: {source}")
            with Image.open(source) as photo:
                rgb = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
            rows[task][side] = gap_score(rgb)
    return rows


def run(output):
    train, held, _ = fixed_sets()
    threshold = max(gap_score(grade(photo, operation)) for photos in train for photo in photos.values()
                    for style, operation in EDITS.items() if style != "contrast")
    rows = []
    for case in held.values():
        for role, photo in pair(BASE / "bg-real-pixel-inputs", case["input"]).items():
            for style, operation in EDITS.items():
                edited = grade(photo, operation)
                rows.append({"case": case["target_pair"], "reference": role, "edit": style, "scores": measure(edited)})
    capture = dast_negatives(BASE / "dast-exposure-prepared-v1")
    results = {}
    for name in VARIANTS:
        detected = sum(row["edit"] == "contrast" and row["scores"][name] > threshold for row in rows)
        wrong_edit = sum(row["edit"] != "contrast" and row["scores"][name] > threshold for row in rows)
        wrong_capture = sum(row["scores"][name] > threshold for row in capture)
        results[name] = {"detected_contrast": detected, "other_edit_false_positives": wrong_edit,
                         "capture_false_positives": wrong_capture, "passed": detected >= 6 and wrong_edit <= 2
                         and wrong_capture <= 3}
    output.mkdir()
    save_report(output / "results.json", {"threshold": threshold, "results": results, "development": rows,
                "capture_negatives": capture, "authored_examples": authored_examples(BASE / "psbench-local-intent-pair-v1"),
                "passed": all(item["passed"] for item in results.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
