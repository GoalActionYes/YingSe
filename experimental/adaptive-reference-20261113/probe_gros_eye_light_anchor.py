"""Test whether portrait eye-area highlights are useful local color anchors."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_physical_blue_global_fit import display, linear
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb,
    digest, finish_spatial, json_to, panel, save,
)

LUMA = np.array([.2126, .7152, .0722])
PARENT = "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260"


def eye_anchor(image, eyes):
    samples = []
    for x, y in eyes:
        patch = image[max(0, y - 4):y + 5, max(0, x - 6):x + 7]
        valid = ((patch >= 8) & (patch <= 247)).all(axis=-1)
        rgb = linear(patch[valid])
        if len(rgb) < 8:
            return None
        rank = np.argsort(rgb @ LUMA, kind="stable")
        samples.append(rgb[rank[-8:]])
    chroma = np.median(np.log(np.concatenate(samples)), axis=0)
    return chroma - chroma.mean()


def adjust(reference, eyes, difference):
    if difference is None or not difference.any():
        return reference, 0.0
    eye_distance = np.linalg.norm(np.subtract(*eyes))
    cx, cy = np.mean(eyes, axis=0)
    y, x = np.ogrid[:reference.shape[0], :reference.shape[1]]
    radius = np.hypot(x - cx, y - (cy + eye_distance / 4)) / eye_distance
    taper = 0.5 + 0.5 * np.cos(np.pi * np.clip(2 * radius - 1, 0, 1))
    active = taper > 0
    adjusted = reference.copy()
    source = linear(reference[active])
    changed = source * np.exp(taper[active, None] * difference)
    scale = (source @ LUMA) / np.maximum(changed @ LUMA, 1e-12)
    changed *= scale[:, None]
    adjusted[active] = display(changed)
    return adjusted, float((changed > 1).any(axis=-1).sum() / active.size)


def run(out):
    root = BASE / "gros-film-b-cross-v2"
    parent = checked_json(root / "results.json", PARENT)
    source = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    labels = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(labels) != source["regions_sha256"]:
        raise ValueError("Changed registered regions")
    coords = json.loads(labels.read_bytes())
    frames = {
        (person, light, state): checked_rgb(root / f"{person}-{light}-{state}.png",
                                             parent["inputs"][f"{person}-{light}"]["hashes"][state])
        for person in GROUPS for light in LIGHTS for state in ("off", "on")
    }
    anchors = {
        key: eye_anchor(image, coords[key[0]][key[1]]["eyes"])
        for key, image in frames.items()
    }
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or parent["strength"] != STRENGTH:
        raise ValueError("Changed spatial model or strength")
    out.mkdir()
    controls = {}
    for person in GROUPS:
        target = frames[person, "cold", "off"]
        anchor = anchors[person, "cold", "off"]
        ref, _ = adjust(target, coords[person]["cold"]["eyes"], None if anchor is None else anchor - anchor)
        frame = finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
        controls[person] = dict(identity=bool(np.array_equal(target, frame)),
                                sha256=save(out / f"{person}-identity.png", frame))

    cases = {}
    for name, row in parent["cases"].items():
        person, donor = name.split("-from-")
        target = frames[person, "cold", "off"]
        target_anchor = anchors[person, "cold", "off"]
        actual, baseline, references, notes, hashes = {}, {}, {}, {}, {}
        for light in LIGHTS:
            actual[light], baseline[light] = {}, {}
            for state in ("off", "on"):
                key = (donor, light, state)
                anchor = anchors[key]
                difference = None if anchor is None or target_anchor is None else target_anchor - anchor
                ref, clip = adjust(frames[key], coords[donor][light]["eyes"], difference)
                stem = f"{light}-{state}"
                rendered = finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
                actual[light][state] = rendered
                baseline[light][state] = checked_rgb(root / f"{name}-13-{stem}.png", row["hashes"][f"13-{stem}"])
                references[stem] = ref
                hashes[stem] = save(out / f"{name}-eye-{stem}.png", rendered)
                notes[stem] = dict(anchor_delta=None if difference is None else difference.tolist(), clip_fraction=clip)

        expected = checked_rgb(root / f"{name}-expected.png", row["expected_sha"])
        zones = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        flatten = lambda levels: {f"{light}-{state}": im for light, tiles in levels.items()
                                  for state, im in tiles.items()}
        board = {"target": {"original": target, "Film B": expected}, "adjusted ref": references,
                 "13": flatten(baseline), "eye": flatten(actual)}
        cases[name] = dict(baseline=score(target, expected, baseline, zones),
                           eye=score(target, expected, actual, zones), notes=notes, hashes=hashes,
                           panel_sha=panel(out / f"{name}-comparison.png", board))
    script = Path(__file__)
    json_to(out / "results.json", dict(
        parent=PARENT, checkpoint=CHECKPOINT, strength=STRENGTH, controls=controls, cases=cases,
        anchors={"-".join(key): None if value is None else value.tolist() for key, value in anchors.items()},
        script_sha=digest(script), protocol_sha=digest(script.with_name("gros-eye-light-anchor-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
