"""Test the prior known-material light replay on separate wood scenes."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked, table
from fetch_mls_linear_pilot import INDEX_SHA, LIGHTS
from probe_fau_illuminant_replay import replay
from probe_fau_two_light_response import encode
from probe_mls_light_response import GROUPS, evaluate, registration
from probe_mls_linear_pilot import read, valid_pixels
from probe_mls_material_response import light_gain
from probe_multilum_chromaticity import differences
from probe_physical_blue_global_fit import save
from spatial_render import finish_spatial


SCENES = ("13", "22")


def prepare(source, specifications, masks, intake, scene, out):
    names = [f"{scene}_{light}.png" for light in LIGHTS]
    raw = [read(source / name, intake["entries"][name]) for name in names]
    before, after = (encode((image.astype(np.uint32) * 4).clip(0, 65535)) for image in raw)
    if before.shape != after.shape:
        raise ValueError("Different capture dimensions")
    height, width = before.shape[:2]
    with Image.open(BytesIO(masks.read(f"masks_preview/{scene}.png"))) as mask:
        if abs((width / height) / (mask.width / mask.height) - 1) >= .002:
            raise ValueError("Mask and capture aspect ratios differ")
        palette = np.array(mask.resize((width, height), Image.Resampling.NEAREST))
    areas = {}
    for name in GROUPS[scene][1]:
        rgb, = [row["mask_colour_rgb"] for row in specifications
                if int(row["scene"]) == int(scene) and row["surface"] == name]
        areas[name] = np.all(palette == np.array(rgb.split(), dtype=int), axis=-1)
    valid = valid_pixels(before, after)
    aligned = registration(before, after)
    expected = finish_spatial(before, after, .8)
    qualification = {name: evaluate(before, before, expected, area, valid, aligned >= .5)
                     for name, area in areas.items()}
    identities = {name: save(out / name, rgb) for name, rgb in zip(names, (before, after), strict=True)}
    return (before, after, expected, valid, areas), {
        "registration": aligned, "qualification": qualification, "inputs_sha256": identities,
        "gain_clipped": [int((image > 16383).any(axis=2).sum()) for image in raw],
    }


def replay_scenes(prepared, results, out):
    measurements = {}
    for scene, (before, after, _, valid, areas) in prepared.items():
        selected = {name: area & valid for name, area in areas.items()}
        measurements[scene] = {name: (differences(before[use], after[use])[0].mean(axis=0), int(use.sum()))
                               for name, use in selected.items()}
    for scene, (before, _, truth, valid, areas) in prepared.items():
        donor, names = GROUPS[scene]
        offsets, weights = zip(*(measurements[donor][name] for name in names), strict=True)
        mean = np.average(offsets, weights=weights, axis=0)
        arms = {"matched": offsets, "crossed": offsets[::-1], "pooled": (mean, mean),
                "zero": (np.zeros(2), np.zeros(2))}
        outside = ~np.logical_or.reduce(list(areas.values()))
        result = {}
        for arm, pairs in arms.items():
            rendered = before.copy()
            for name, offset in zip(names, pairs, strict=True):
                area = areas[name]
                rendered[area] = replay(before, light_gain(offset), area)[0][area]
            np.testing.assert_array_equal(rendered[outside], before[outside])
            if arm == "zero":
                np.testing.assert_array_equal(rendered, before)
            result[arm] = {"sha256": save(out / f"{scene}-{arm}.png", rendered),
                           "regions": {name: evaluate(before, rendered, truth, area, valid, True)
                                       for name, area in areas.items()}}
        gates = {}
        for name in names:
            matched = result["matched"]["regions"][name]
            improvements = {}
            for arm in ("pooled", "crossed"):
                error = result[arm]["regions"][name]["error"]
                improvements[arm] = 1 - matched["error"] / error if error else None
            gates[name] = {"improvements": improvements,
                           "passed": matched["lamp_pass"] and all(v is not None and v >= .15 for v in improvements.values())}
        results[scene].update(arms=result, gates=gates, donor=donor,
                              observed={name: row[0].tolist() for name, row in measurements[scene].items()})


def run(source, material, out):
    receipt = json.loads((source / "intake.json").read_bytes())
    if receipt["index_sha256"] != INDEX_SHA:
        raise ValueError("Unexpected archive index")
    with checked(material, "surfaces.zip") as spectra:
        specifications = table(spectra, "surfaces/mask_to_surface.csv")
    out.mkdir()
    prepared, results = {}, {}
    with checked(material, "masks_preview.zip") as masks:
        for scene in SCENES:
            prepared[scene], results[scene] = prepare(source, specifications, masks, receipt, scene, out)
    qualified = all(row["eligible"] for case in results.values() for row in case["qualification"].values())
    if qualified:
        replay_scenes(prepared, results, out)
    passed = qualified and all(row["passed"] for case in results.values() for row in case["gates"].values())
    report = {"source": receipt, "cases": results, "qualified": qualified, "passed": passed}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "material", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
