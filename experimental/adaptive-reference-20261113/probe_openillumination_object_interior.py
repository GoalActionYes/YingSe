"""Measure reference-object lighting while holding its surroundings fixed."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from probe_openillumination_visible_emitters import (
    BASE, CHECKPOINT, INPUTS, PARENT, SCENES, STATES, STRENGTH, SpatialEngine,
    attribution, checked_json, checked_rgb, digest, finish_spatial, grade, json_to,
    panel, render_style, save, score, style_transforms,
)

ARMS = {"add_object_light": ("001", "013"), "remove_object_light": ("013", "001")}


def splice(base, donor, weight):
    mixed = np.rint(base + (donor.astype(np.float32) - base) * weight[..., None]).astype(np.uint8)
    np.testing.assert_array_equal(mixed[weight == 0], base[weight == 0])
    np.testing.assert_array_equal(mixed[weight == 1], donor[weight == 1])
    if np.array_equal(mixed, base):
        raise ValueError("Object intervention made no change")
    return mixed


def run(out):
    source = BASE / "openillumination-fixed-prepared-v1"
    previous = BASE / "openillumination-fixed-cross-v1"
    prepared = checked_json(source / "inputs.json", INPUTS)
    frozen = checked_json(previous / "results.json", PARENT)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed baseline model or strength")
    film = style_transforms(BASE)["film_b"]
    out.mkdir()
    photos, objects, guides, inputs = {}, {}, {}, {}
    for scene in SCENES:
        record = prepared["objects"][scene]
        photos[scene] = frames = {key: checked_rgb(source / f"{scene}-{key}.png", record["frames"][key])
                                  for key in ("001", "013")}
        path = source / f"{scene}-interior.png"
        if digest(path) != record["mask_sha256"]:
            raise ValueError("Changed object mask")
        with Image.open(path) as image:
            objects[scene] = region = np.array(image) > 0
        weight = np.minimum(cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / 4, 1)
        if not (weight == 1).any() or not np.array_equal(weight > 0, region):
            raise ValueError("Intervention does not cover the prepared region")
        mask_hash = save(out / f"{scene}-weight.png", np.rint(weight * 255).astype(np.uint8))
        guides[scene] = {}
        inputs[scene] = dict(weight=mask_hash, pixel_count=int(region.sum()), core_count=int((weight == 1).sum()),
                             references={})
        for arm, (base, donor) in ARMS.items():
            photo = splice(frames[base], frames[donor], weight)
            guides[scene][arm] = {"off": photo, "on": render_style(photo, film)}
            for state, pixels in guides[scene][arm].items():
                key = f"{arm}-{state}"
                inputs[scene]["references"][key] = save(out / f"{scene}-guide-{key}.png", pixels)

    cases = {}
    for target_name, donor in permutations(SCENES, 2):
        name = f"{target_name}-from-{donor}"
        baseline = frozen["cases"][name]
        target = photos[target_name]["001"]
        expected = finish_spatial(target, render_style(target, film), STRENGTH)
        region = objects[target_name]
        areas = {"whole": np.ones_like(region), "object": region}
        original = {light: {state: checked_rgb(previous / f"{name}-{tag}-{state}.png",
                                               baseline["hashes"][f"{tag}-{state}"]) for state in STATES}
                    for light, tag in (("cold", "001"), ("warm", "013"))}
        if score(target, expected, original, areas) != baseline["scores"]:
            raise ValueError("Changed frozen scores")
        rendered = {arm: {state: grade(engine, target, ref) for state, ref in frames.items()}
                    for arm, frames in guides[donor].items()}
        add, remove = rendered["add_object_light"], rendered["remove_object_light"]
        effects = {}
        for state in STATES:
            effects[state] = {}
            for zone, where in areas.items():
                result = attribution(original["cold"][state], add[state], remove[state], original["warm"][state], where)
                result["object_only"] = result.pop("emitter_only")
                result["object_projection"] = result.pop("emitter_projection")
                effects[state][zone] = result
        cases[name] = dict(effects=effects,
                           scores={arm: score(target, expected, {"cold": original["cold"], "warm": variants}, areas)
                                   for arm, variants in rendered.items()},
                           hashes={f"{arm}-{state}": save(out / f"{name}-{arm}-{state}.png", image)
                                   for arm, images in rendered.items() for state, image in images.items()},
                           board=panel(out / f"{name}-comparison.png", {
                               "target": {"original": target, "expected": expected}, "001": original["cold"],
                               "013": original["warm"], **rendered,
                           }))
    protocol = Path(__file__).with_name("openillumination-object-interior-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, inputs=INPUTS, checkpoint=CHECKPOINT, strength=STRENGTH,
                                       protocol=digest(protocol), regions=inputs, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
