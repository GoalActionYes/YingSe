"""Replay a material's held-out MLS light response from unpaired regional statistics."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageFilter

from audit_mls_material_identity import checked, table
from audit_mls_multiscene_intake import SCENES
from fetch_mls_linear_pilot import LIGHTS
from independent_coco import digest
from probe_fau_illuminant_replay import replay
from probe_fau_two_light_response import encode
from probe_mls_linear_pilot import read
from probe_mls_material_response import light_gain
from probe_multilum_chromaticity import linear_rgb


OFFSETS = [(dy, dx) for dy in (-6, 0, 6) for dx in (-6, 0, 6) if dy or dx]
LEVELS = (.10, .50, .90)


def summary(frame, region):
    pixels = frame[region]
    good = ((pixels >= 16) & (pixels <= 239)).all(axis=-1)
    selected = pixels[good]
    if not len(selected):
        raise ValueError("No pixels in the exposure interval")
    log_values = np.log(linear_rgb(selected))
    color = log_values[:, [0, 2]] - log_values[:, 1:2]
    return np.quantile(color, LEVELS, axis=0), int(len(selected))


def prepare(scene, material, source, receipt, mask, rows):
    names = [f"{scene}_{light}.png" for light in LIGHTS]
    raw = [read(source / name, receipt[name]) for name in names]
    frames = [encode((image.astype(np.uint32) * 4).clip(0, 65535)) for image in raw]
    with Image.open(BytesIO(mask)) as image:
        rgb = np.asarray(image)
    label, = [
        np.fromstring(row["mask_colour_rgb"], dtype=int, sep=" ")
        for row in rows if int(row["scene"]) == int(scene) and row["surface"] == material
    ]
    region = np.all(rgb == label, axis=-1)
    interior = np.asarray(Image.fromarray(region.astype(np.uint8) * 255)
                          .filter(ImageFilter.MinFilter(17))) > 0
    interior[:6] = interior[-6:] = False
    interior[:, :6] = interior[:, -6:] = False
    first, second = [summary(frame, interior) for frame in frames]
    measured = second[0] - first[0]
    np.testing.assert_array_equal(first[0] - summary(frames[0], interior)[0], np.zeros((3, 2)))
    variations = []
    for dy, dx in OFFSETS:
        shifted = np.roll(interior, (dy, dx), axis=(0, 1))
        for i, frame in enumerate(frames):
            alternate, count = summary(frame, shifted)
            saved = (first, second)[i][0]
            variations.append({"light": LIGHTS[i], "shift": [dy, dx], "valid": count,
                               "delta": float(np.linalg.norm(alternate[1] - saved[1]))})
    maximum = max(shift["delta"] for shift in variations)
    observed = float(np.linalg.norm(measured[1]))
    size = int(interior.sum())
    qualified = (min(first[1], second[1], *(row["valid"] for row in variations)) >= max(256, size * .5)
                 and maximum <= .02 and maximum <= .10 * observed)
    stats = {"source_sha256": {name: receipt[name]["sha256"] for name in names},
             "interior_pixels": size, "valid_pixels": [first[1], second[1]],
             "observed_quantile_response": measured.tolist(), "maximum_shift_delta": maximum,
             "observed_norm": observed, "offsets": variations, "qualified": qualified}
    return frames, interior, first[0], measured, stats


def score(frame, region, reference, baseline):
    predicted, count = summary(frame, region)
    change = predicted - baseline
    error = np.linalg.norm(change - reference, axis=1)
    power = float(reference[1] @ reference[1])
    return {"predicted": change.tolist(), "valid_pixels": count, "quantile_errors": error.tolist(),
            "projection": float(change[1] @ reference[1] / power) if power else None}


def run(old, added, labels, qualification, out):
    parent = json.loads(qualification.read_bytes())
    assert parent["control_panel_valid"]
    inputs = {path: json.loads((path / "intake.json").read_bytes())["entries"] for path in (old, added)}
    with checked(labels, "surfaces.zip") as spectra:
        rows = table(spectra, "surfaces/mask_to_surface.csv")
    out.mkdir()
    prepared = {}
    with ZipFile(added / "masks_16bit_color.zip") as masks:
        assert masks.testzip() is None
        names = {Path(entry).name: entry for entry in masks.namelist() if entry.endswith(".png")}
        for scene, material in SCENES.items():
            path = old if scene in ("02", "03") else added
            prepared[scene] = prepare(scene, material, path, inputs[path], masks.read(names[f"{scene}.png"]), rows)
            assert prepared[scene][-1]["source_sha256"] == parent["scenes"][scene]["files"]

    output = {}
    for scene, material in SCENES.items():
        frames, region, base, measured, stats = prepared[scene]
        same = [key for key in SCENES if key != scene and SCENES[key] == material]
        other = [key for key in SCENES if SCENES[key] != material]
        matched, wrong = (np.mean([prepared[key][3][1] for key in donors], axis=0) for donors in (same, other))
        offsets = {"matched": matched, "wrong": wrong, "mixed": (matched + wrong) / 2,
                   "zero": np.zeros(2), "known_response": measured[1]}
        target = measured * .8
        arms = {}
        for arm, offset in offsets.items():
            rendered, clipped = replay(frames[0], light_gain(offset), region)
            np.testing.assert_array_equal(rendered[~region], frames[0][~region])
            if arm == "zero":
                np.testing.assert_array_equal(rendered, frames[0])
            path = out / f"{scene}-{arm}.png"
            Image.fromarray(rendered).save(path)
            arms[arm] = score(rendered, region, target, base) | {"clipped": clipped, "sha256": digest(path)}
        selected = arms["matched"]
        errors = selected["quantile_errors"]
        mixed = arms["mixed"]["quantile_errors"]
        wins = all(errors[1] <= .85 * arms[arm]["quantile_errors"][1] and errors[1] < arms[arm]["quantile_errors"][1]
                   for arm in ("wrong", "mixed", "zero"))
        noninferior_tails = all(errors[i] <= mixed[i] + .02 for i in (0, 2))
        projected = selected["projection"] is not None and .75 <= selected["projection"] <= 1.25
        output[scene] = {"material": material, "same_donors": same, "other_donors": other,
                         "measurement": stats, "arms": arms, "passes": wins and noninferior_tails and projected}
    report = {"source_sha256": digest(Path(__file__)), "qualification_sha256": digest(qualification),
              "cases": output, "passed": all(row["measurement"]["qualified"] and row["passes"] for row in output.values())}
    (out / "results.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "added", "labels", "qualification", "out"):
        parser.add_argument(arg, type=Path)
    values = parser.parse_args()
    run(values.old, values.added, values.labels, values.qualification, values.out)
