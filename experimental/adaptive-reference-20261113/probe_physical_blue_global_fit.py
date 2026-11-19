"""Compare offline global-color fits of physical blue and intentional digital blue."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CAPTURE_SHA = "ae4bdcfbaec10bf43c11b882138745ef02f86f58011d6102a43a666630e83789"
DIGITAL_SHA = "de5de04d6eaf9debd53626dfc4e16a7d9629343cd715fab885577d8141157570"
SCENES = ("Scene1_a", "Scene2_a")


def verified(path, expected):
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Changed input: {path}")
    return path


def pixels(path, expected):
    with Image.open(verified(path, expected)) as image:
        return np.array(image.convert("RGB"))


def linear(rgb):
    value = rgb.astype(float) / 255
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def display(rgb):
    value = rgb.clip(0, 1)
    encoded = np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)
    return np.rint(encoded * 255).astype(np.uint8)


def inside(rgb):
    return np.all((rgb >= 4) & (rgb <= 251), axis=2)


def score(picture, wanted, mask):
    error = np.abs(picture.astype(np.int16) - wanted).mean(axis=2)
    return {"rgb8_mae": float(error[mask].mean()), "pixel_mean_channel_error_p95": float(np.percentile(error[mask], 95))}


def fit(source, wanted):
    original, desired = linear(source), linear(wanted)
    usable = inside(source) & inside(wanted)
    y, x = np.ogrid[:source.shape[0], :source.shape[1]]
    folds = (x // 32 + 2 * (y // 32)) % 4
    rendered = np.empty_like(source)
    audit = []
    for fold in range(4):
        held = folds == fold
        training = usable & ~held
        fit_count, test_count = int(training.sum()), int((held & usable).sum())
        matrix, _, rank, _ = np.linalg.lstsq(original[training], desired[training], rcond=None)
        if min(fit_count, test_count) < 512 or rank != 3:
            raise ValueError(f"Unusable fold {fold}: training={fit_count} test={test_count} rank={rank}")
        rendered[held] = display(original[held] @ matrix)
        audit.append({"training": fit_count, "test": test_count, "matrix": matrix.tolist()})
    return rendered, {"folds": audit, "valid_fraction": float(usable.mean()),
                      "eligible": score(rendered, wanted, usable),
                      "whole": score(rendered, wanted, np.ones(usable.shape, dtype=bool))}


def save(path, rgb):
    Image.fromarray(rgb).save(path)
    return sha256(path.read_bytes()).hexdigest()


def panel(path, rows):
    board = Image.new("RGB", (1050, 430), "#222730")
    draw = ImageDraw.Draw(board)
    for index, (name, rgb) in enumerate(rows.items()):
        x, y = 8 + index % 5 * 210, 6 + index // 5 * 212
        tile = Image.fromarray(rgb)
        tile.thumbnail((194, 180))
        draw.text((x, y), name, fill="white")
        board.paste(tile, (x + (194 - tile.width)//2, y + 22))
    board.save(path)
    return sha256(path.read_bytes()).hexdigest()


def run(base, out):
    captures = base / "chromamill-capture-cross-v1"
    digitally = base / "cti-selective-correction-v1"
    original = json.loads(verified(captures / "results.json", CAPTURE_SHA).read_bytes())
    edits = json.loads(verified(digitally / "results.json", DIGITAL_SHA).read_bytes())
    out.mkdir()
    results = {}
    for scene, level in product(SCENES, (213, 241)):
        donor_case = SCENES[1] if scene == SCENES[0] else SCENES[0]
        choice = edits["digital"][donor_case][str(level)]
        if not choice["qualified"]:
            raise ValueError("Known digital control is no longer eligible")
        tag = f"{scene}-{level}"
        records, pictures = {}, {}
        for state in ("off", "on"):
            white, blue = (pixels(captures / f"{scene}-{light}-{level}-{state}.png",
                                  original["prepared"][scene][f"{light}-{level}"]["hashes"][state])
                           for light in ("white", "blue"))
            digital = pixels(digitally / f"{donor_case}-{level}-{state}-reference.png",
                             choice["states"][state]["hashes"]["reference"])
            row = {"mean_matched_physical_vs_digital": score(digital, blue, inside(digital) & inside(blue))}
            pictures[f"{state} / White"] = white
            for name, desired in (("physical", blue), ("digital", digital)):
                candidate, audit = fit(white, desired)
                row[name] = audit | {"sha256": save(out / f"{tag}-{state}-{name}.png", candidate)}
                pictures[f"{state} / {name}"] = desired
                pictures[f"{state} / {name} fit"] = candidate
            records[state] = row
        off = records["off"]
        physical, digital = (off[name]["eligible"] for name in ("physical", "digital"))
        absolute = physical["rgb8_mae"] - digital["rgb8_mae"]
        control = digital["rgb8_mae"] <= 1 and digital["pixel_mean_channel_error_p95"] <= 3
        separation = (physical["rgb8_mae"] > 3 and physical["pixel_mean_channel_error_p95"] > 8 and absolute >= 2)
        results[tag] = {"states": records, "control_pass": control, "physical_separation": separation,
                        "panel_sha256": panel(out / f"{tag}-comparison.png", pictures)}
    report = {"parent_sha256": {"capture": CAPTURE_SHA, "digital": DIGITAL_SHA}, "cases": results,
              "all_off_digital_controls_pass": all(row["control_pass"] for row in results.values()),
              "all_off_physical_separated": all(row["physical_separation"] for row in results.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
