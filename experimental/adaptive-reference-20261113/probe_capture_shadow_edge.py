"""Test a same-material shadow-edge proxy as a two-image grading control."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_beyond_rgb_matched_cast import compare
from probe_capture_tile_tone_color import MID_SHA, STATES, acquire, classify, describe, render as render_mit
from probe_multilum_gray_probe import chroma, linear, valid
from probe_gros_render_intent_projection import grade
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)
from probe_wildrelight_measured_gain import INPUT_SHA, PARENT, REGIONS, areas_for, gains
from probe_wildrelight_wall_cross import LUMA, SCENES, image, prepared_rgb

ARMS = {"flat": describe}
STEP = 12


def shadow_pairs(rgb):
    values = linear(rgb)
    light = np.log(np.maximum(values @ LUMA, 1e-6))
    color = chroma(rgb)
    good = valid(rgb)
    outputs = []
    for axis in (0, 1):
        luma, chromaticity, usable = (np.moveaxis(field, axis, 0) for field in (light, color, good))
        distance = STEP + 2
        left, right = luma[2:-distance], luma[distance:-2]
        step = right - left
        flat = ((np.abs(left-luma[:-STEP-4]) <= np.log(1.12))
                & (np.abs(left-luma[4:-STEP]) <= np.log(1.12))
                & (np.abs(right-luma[STEP:-4]) <= np.log(1.12))
                & (np.abs(right-luma[STEP+4:]) <= np.log(1.12)))
        delta = chromaticity[distance:-2] - chromaticity[2:-distance]
        supported = (usable[2:-distance] & usable[distance:-2] & flat
                     & (np.abs(step) >= np.log(1.35)) & (np.linalg.norm(delta, axis=-1) <= .26))
        signed = delta * np.sign(step)[..., None]
        outputs.append(signed[supported])
    return np.concatenate(outputs, axis=0)


def shadow(rgb):
    scaled = Image.fromarray(rgb)
    scaled.thumbnail((384, 384), Image.Resampling.BILINEAR)
    values = shadow_pairs(np.asarray(scaled))
    if len(values) < 128:
        return None
    return np.quantile(values, [.5, .8], axis=0).reshape(-1)


def synthetic():
    data = np.empty((100, 100, 3), dtype=np.uint8)
    data[:, :50] = (80, 80, 80)
    data[:, 50:] = (135, 135, 135)
    neutral = shadow_pairs(data)
    if len(neutral) < 128 or not np.all(neutral == 0):
        raise ValueError("Achromatic shadow control failed")
    same = np.broadcast_to(data[0, 0], data.shape)
    if len(shadow_pairs(same)):
        raise ValueError("Flat image produced shadow edges")


def fetch_wild():
    source = BASE / "wildrelight-wall-prepared-v1"
    raw = BASE / "wildrelight-small-v1"
    frozen = checked_json(source / "inputs.json", INPUT_SHA)
    photos, groups, rebalance = {}, {}, {}
    for scene, entry in frozen["scenes"].items():
        a, b = (checked_rgb(source / f"{scene}-{time}.png", entry["hashes"][time]) for time in ("time0", "time5"))
        key = f"small-aligned/{scene}/photo/time0_hdr.exr"
        if digest(raw / key) != frozen["inputs"][key]:
            raise ValueError("Raw source changed")
        hdr = image(raw / key)
        np.testing.assert_array_equal(prepared_rgb(hdr, entry["scale"]), a)
        change = 1 / gains(entry)["rgb"]
        photos[scene] = hdr, entry["scale"]
        rebalance[scene] = change
        groups[scene] = dict(base=a, captured=b, edited=prepared_rgb(hdr * change, entry["scale"]))
    return groups, photos, rebalance


def render_wild(groups, photos, rebalance, choices, out):
    field = checked_json(Path(__file__).with_name("wildrelight-wall-cross-regions.json"), REGIONS)
    parent = BASE / "wildrelight-wall-cross-v1"
    frozen = checked_json(parent / "results.json", PARENT)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Renderer changed")
    cases = {}
    for name, donor in (SCENES, SCENES[::-1]):
        target = groups[name]["base"]
        hdr, scale = photos[name]
        expected = finish_spatial(target, prepared_rgb(hdr * rebalance[donor], scale), STRENGTH)
        areas = areas_for(target.shape[:2], field[name]["neutral"])
        locked = frozen["cases"][name]["hashes"]
        frames = {
            role: checked_rgb(parent / f"{name}-from-{donor}-{time}-off.png", locked[f"{time}-off"])
            for role, time in (("base", "time0"), ("captured", "time5"))
        }
        frames["edited"] = grade(engine, target, groups[donor]["edited"])
        renders = {"13": frames} | {
            arm: {label: (target if info[donor][label]["capture"] is True else frame)
                  for label, frame in frames.items()} for arm, info in choices.items()
        }
        def scores(images):
            values = {label: compare(target, expected, *images.values(), area) for label, area in areas.items()}
            return dict(areas=values, passed=all(row["both_pass"] and row["style"]["off_matches_original_mae"] <= 3
                                                for row in values.values()))
        cases[name] = dict(scores={arm: scores(images) for arm, images in renders.items()},
                           row_hashes={f"{arm}-{label}": save(out / f"{name}-{arm}-{label}.png", frame)
                                       for arm, images in renders.items() for label, frame in images.items()},
                           board=panel(out / f"{name}-comparison.png", {
                               "target": {"original": target, "expected edit": expected}, **renders,
                           }))
    return cases


def run(out):
    synthetic()
    groups, mid = acquire()
    wild, hdr, gain = fetch_wild()
    groups["wild"] = wild
    arms = ARMS | {"edges": shadow}
    choices = {}
    totals = {}
    for arm, features in arms.items():
        inputs = {family: {name: {key: features(photo) for key, photo in row.items()}
                           for name, row in records.items()} for family, records in groups.items()}
        choices[arm] = {family: classify(inputs, family) for family in inputs}
        totals[arm] = {}
        for family, records in choices[arm].items():
            totals[arm][family] = {key: {
                "capture": sum(row[key]["capture"] is True for row in records.values()),
                "missing": sum(row[key]["capture"] is None for row in records.values()),
                "total": len(records),
            } for key in STATES}
    out.mkdir()
    cases = {}
    for arm in arms:
        folder = out / arm
        folder.mkdir()
        cases[arm] = render_mit(mid, choices[arm]["mid"]["copyroom"], folder)
    wild_results = render_wild(wild, hdr, gain, {arm: values["wild"] for arm, values in choices.items()}, out)
    protocol = Path(__file__).with_name("capture-shadow-edge-protocol.md")
    json_to(out / "results.json", dict(mid=MID_SHA, wild=INPUT_SHA, archive=PARENT, regions=REGIONS,
             protocol=digest(protocol), totals=totals, choices=choices, mid_scores=cases, wild_scores=wild_results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
