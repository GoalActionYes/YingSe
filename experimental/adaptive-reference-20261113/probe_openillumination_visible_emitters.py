"""Isolate direct emitter pixels from simultaneous reference appearance changes."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import score
from probe_gros_render_intent_projection import grade
from probe_openillumination_fixed_camera import SCENES
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

INPUTS = "ac336c1757901c9a1f77a14eb656bf94800fb4bd5fb65b7d69d693f0b5653f3f"
PARENT = "6b9f2c7af57709c4f04b236a3a798217777f17a85724121fd9bb32beefdf0d02"
STATES = ("off", "on")
ARMS = {"add_emitters": ("001", "013"), "remove_emitters": ("013", "001")}


def dilate(mask, radius):
    shape = (2 * radius + 1,) * 2
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)) > 0


def emitter_mask(photos, objects):
    hits = []
    for frames in photos.values():
        dark, bright = (frames[key].astype(np.int16) for key in ("001", "013"))
        hits.append((bright.min(axis=-1) >= 230) & ((bright - dark).min(axis=-1) >= 60))
    consensus = np.count_nonzero(hits, axis=0) >= 2
    forbidden = dilate(np.logical_or.reduce(list(objects.values())), 16)
    expanded = dilate(consensus, 5)
    selected = expanded & ~forbidden
    fraction = float(selected.mean())
    if not .001 <= fraction <= .10 or (selected & forbidden).any():
        raise ValueError("Emitter mask failed support or object protection")
    return selected, int(consensus.sum()), int((expanded & forbidden).sum())


def swap(baseline, donor, positions):
    output = baseline.copy()
    output[positions] = donor[positions]
    np.testing.assert_array_equal(output[~positions], baseline[~positions])
    np.testing.assert_array_equal(output[positions], donor[positions])
    return output


def attribution(base, added, removed, full, region):
    base, added, removed, full = [img[region].astype(np.float64) for img in (base, added, removed, full)]
    lamp_at_base, lamp_at_full = added - base, full - removed
    emitter = (lamp_at_base + lamp_at_full) / 2
    rest = ((removed - base) + (full - added)) / 2
    total = full - base
    np.testing.assert_allclose(emitter + rest, total, atol=0, rtol=0)
    energy = float((total**2).sum())
    return dict(full=float(np.abs(total).mean()), emitter_only=float(np.abs(lamp_at_base).mean()),
                after_removal=float(np.abs(removed-base).mean()), interaction=float(np.abs(lamp_at_full-lamp_at_base).mean()),
                emitter_projection=float((total*emitter).sum() / energy) if energy else None)


def run(out):
    source = BASE / "openillumination-fixed-prepared-v1"
    previous = BASE / "openillumination-fixed-cross-v1"
    prepared = checked_json(source / "inputs.json", INPUTS)
    frozen = checked_json(previous / "results.json", PARENT)
    photos, objects = {}, {}
    for scene in SCENES:
        record = prepared["objects"][scene]
        photos[scene] = {key: checked_rgb(source / f"{scene}-{key}.png", record["frames"][key])
                         for key in ("001", "013")}
        path = source / f"{scene}-interior.png"
        if digest(path) != record["mask_sha256"]:
            raise ValueError("Changed object mask")
        with Image.open(path) as im:
            objects[scene] = np.array(im) > 0
    mask, core, excluded = emitter_mask(photos, objects)
    if np.any(mask & np.logical_or.reduce(list(objects.values()))):
        raise ValueError("Emitter intervention overlaps scoring objects")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed frozen baseline")
    film = style_transforms(BASE)["film_b"]
    out.mkdir()
    mask_hash = save(out / "emitter-mask.png", mask.astype(np.uint8) * 255)
    guides, inputs = {}, {}
    for scene, frames in photos.items():
        guides[scene], inputs[scene] = {}, {}
        for arm, (other, lamps) in ARMS.items():
            photo = swap(frames[other], frames[lamps], mask)
            if np.array_equal(photo, frames[other]):
                raise ValueError("Emitter swap made no change")
            guides[scene][arm] = {"off": photo, "on": render_style(photo, film)}
            for state, pixels in guides[scene][arm].items():
                inputs[scene][f"{arm}-{state}"] = save(out / f"{scene}-guide-{arm}-{state}.png", pixels)
    cases = {}
    for target_name, donor in permutations(SCENES, 2):
        name = f"{target_name}-from-{donor}"
        parent = frozen["cases"][name]
        target = photos[target_name]["001"]
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        field = objects[target_name]
        areas = {"whole": np.ones_like(field), "object": field}
        original = {light: {state: checked_rgb(previous / f"{name}-{tag}-{state}.png",
                                              parent["hashes"][f"{tag}-{state}"]) for state in STATES}
                    for light, tag in (("cold", "001"), ("warm", "013"))}
        if score(target, expected, original, areas) != parent["scores"]:
            raise ValueError("Frozen baseline stats differ")
        rendered = {arm: {state: grade(engine, target, ref) for state, ref in frames.items()}
                    for arm, frames in guides[donor].items()}
        scores = {arm: score(target, expected, {"cold": original["cold"], "warm": variants}, areas)
                  for arm, variants in rendered.items()}
        contributions = {state: {zone: attribution(original["cold"][state], rendered["add_emitters"][state],
                                                     rendered["remove_emitters"][state], original["warm"][state], where)
                                  for zone, where in areas.items()} for state in STATES}
        cases[name] = dict(scores=scores, contributions=contributions,
                           hashes={f"{arm}-{state}": save(out / f"{name}-{arm}-{state}.png", image)
                                   for arm, images in rendered.items() for state, image in images.items()},
                           board=panel(out / f"{name}-comparison.png", {
                               "target": {"original": target, "expected": expected}, "001": original["cold"],
                               "013": original["warm"], **rendered,
                           }))
    protocol = Path(__file__).with_name("openillumination-visible-emitters-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, inputs=INPUTS, checkpoint=CHECKPOINT, strength=STRENGTH,
                                       protocol=digest(protocol), mask=dict(pixels=int(mask.sum()), fraction=float(mask.mean()),
                                           common_core=core, excluded=excluded, sha=mask_hash), references=inputs, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
