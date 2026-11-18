"""Measure what a supplied reference subject region adds to a fixed grade probe."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_bg_real_crossed_grade_training import (BASE, ROOT, PARENT, PARENT_SHA, FIT_SHA, HELD_SHA, EDITS,
    describe, digest, feature, fit, grade, headroom, mask_for, pair, panel, read, scores, target_regions)
from probe_bg_real_pixel_pilot import interior


BASELINE_SHA = "cbd75c0361b4a52068e9527dfb5a886c919843daf320434113dba2e6951de47b"


def core_mask(folder, entry, shape):
    path = folder/entry["name"]
    if digest(path) != entry["sha256"]:
        raise ValueError("Frozen reference mask changed")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.shape != shape:
        raise ValueError("Reference mask dimensions changed")
    mask = interior(pixels >= 128)
    if mask.sum() < 256:
        raise ValueError("Reference core lacks measurement coverage")
    return mask


def train(inputs, inventory):
    pictures = {k: pair(inputs, inventory[k]) for k in sorted(inventory)}
    target_descriptors = [describe(value["C1"]) for value in pictures.values()]
    predictors = {"crossed": [], "subject": []}
    labels = []
    for i, (key, photos) in enumerate(pictures.items()):
        target = target_descriptors[(i+1)%len(pictures)]
        core = core_mask(inputs, inventory[key]["mask"], photos["C1"].shape[:2])
        for coefficients in EDITS.values():
            for image in photos.values():
                reference = grade(image, coefficients)
                predictors["crossed"].append(feature(target, describe(reference)))
                predictors["subject"].append(feature(target, describe(reference[core])))
                labels.append(coefficients)
    if len(labels) != 96:
        raise ValueError("Unexpected training budget")
    return {arm: fit(x, labels) for arm, x in predictors.items()}


def run(output):
    train_folder = BASE/"bg-real-color-directions-inputs"
    held_folder = BASE/"bg-real-pixel-inputs"
    train_path = train_folder/"intake.json"
    held_path = ROOT/"docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    target_path = PARENT/"results.json"
    base_path = ROOT/"docs/research/2026-11-19-bg-real-crossed-grade-training-evidence.json"
    if tuple(map(digest, (train_path, held_path, target_path, base_path))) != (FIT_SHA, HELD_SHA, PARENT_SHA, BASELINE_SHA):
        raise ValueError("Frozen training or test evidence changed")
    models = train(train_folder, json.loads(train_path.read_bytes()))
    held = json.loads(held_path.read_bytes())["cases"]
    targets = {r["pair"]:r for r in json.loads(target_path.read_bytes())["cases"]}
    frozen = json.loads(base_path.read_bytes())["cases"]
    output.mkdir()
    results, worst = {}, 0.
    for case in held.values():
        key = case["target_pair"]
        entry = targets[key]
        original = read(entry["input_file"])
        canvas = headroom(original)
        target = describe(canvas)
        human = mask_for(entry, key.split("-")[0])
        neutral = target_regions(original, human)["neutral_proxy"] & human
        expected = {style: grade(canvas, coefficients, .8) for style, coefficients in EDITS.items()}
        images = pair(held_folder, case["input"])
        core = core_mask(held_folder, case["input"]["mask"], images["C1"].shape[:2])
        record, panels = {}, {"prescribed edit": expected}
        for kind, (center, scale, intercept, weights) in models.items():
            renders, estimates = {}, {}
            for role, image in images.items():
                renders[role], estimates[role] = {}, {}
                for style, coefficients in EDITS.items():
                    look = grade(image, coefficients)
                    observation = describe(look[core] if kind == "subject" else look)
                    prediction = ((feature(target, observation)-center)/scale) @ weights + intercept
                    estimated = np.clip(prediction, [-1, 0], [1, 1])
                    renders[role][style] = grade(canvas, estimated, .8)
                    estimates[role][style] = estimated.tolist()
                panels[f"{kind} / {role}"] = renders[role]
            record[kind] = {**scores(renders, expected, human, neutral), "estimated": estimates}
        before = BASE/"bg-real-crossed-grade-training"/f"{key}.png"
        if digest(before) != frozen[key]["panel_sha256"]:
            raise ValueError("Frozen crossed result changed")
        with Image.open(before) as original_panel:
            image = np.asarray(original_panel.convert("RGB"))
        for j, (role, images_) in enumerate((v for v in panels.items() if v[0].startswith("crossed /")),1):
            for i, pixels in enumerate(images_.values()):
                x, y = 8+i*264, 26+280*(j+2)
                np.testing.assert_array_equal(pixels, image[y:y+256, x:x+256])
        for style in EDITS:
            for role in images:
                newer, older = (record[arm]["styles"][style]["response"][role]["target_mae"]
                                for arm in ("subject", "crossed"))
                worst = max(worst, newer-older)
        path = output/f"{key}.png"
        panel(path, panels)
        results[key] = {**record, "reference_support": int(core.sum()), "panel_sha256": digest(path)}
    totals = {arm: sum(int(row[arm]["passed"]) for row in results.values()) for arm in models}
    values = {"cases": results, "totals": totals, "worst_regression": worst,
              "passed": totals["subject"] >= 3 and totals["subject"] > totals["crossed"] and worst <= 2.,
              "prior_sha256": BASELINE_SHA, "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("bg-real-crossed-grade-subject-protocol.md"))}
    (output/"results.json").write_text(json.dumps(values, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
