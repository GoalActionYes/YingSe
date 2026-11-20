"""Route signed reference-region edit responses to corresponding target regions."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from probe_gros_bounded_material_field import support
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb,
    digest, finish_spatial, json_to, panel, save,
)


def region_boxes(coords, person, light):
    frame = coords[person][light]
    return {key: frame[key] for key in ("cheek_left", "cheek_right")} | coords["screen"]


def crop(image, box):
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def composite(off, on, routed, weights):
    previous = on.astype(np.int16) - off
    total = weights.sum(axis=-1)
    values = off + previous * np.maximum(1 - total, 0)[..., None]
    values += (weights[..., None] * routed).sum(axis=-2) / np.maximum(total, 1)[..., None]
    return np.rint(np.clip(values, 0, 255)).astype(np.uint8)


def run(out, bounded_sha):
    bounded = BASE / "gros-bounded-material-field-v3"
    material = BASE / "gros-material-capture-field-v1"
    root = BASE / "gros-film-b-cross-v2"
    limits = checked_json(bounded / "results.json", bounded_sha)
    parent = checked_json(material / "results.json", limits["parent"])
    film = checked_json(root / "results.json", parent["parent"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed reference and target region identities")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed checkpoint")
    cases = {}
    for name, case in film["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{name}-expected.png", case["expected_sha"])
        boxes = region_boxes(coords, person, "cold")
        weights = np.stack([support(target.shape[:2], [x]) for x in boxes.values()], axis=-1)
        areas = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        old, candidate = {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                guides = {state: checked_rgb(root / f"{donor}-cold-{state}.png",
                                              film["inputs"][f"{donor}-cold"]["hashes"][state])
                          for state in ("off", "on")}
                full = {state: checked_rgb(root / f"{name}-13-cold-{state}.png",
                                            case["hashes"][f"13-cold-{state}"]) for state in ("off", "on")}
            else:
                guides = {state: checked_rgb(bounded / f"{donor}-bounded-{state}.png",
                                              limits["prepared"][donor]["hashes"][state]) for state in ("off", "on")}
                full = {state: checked_rgb(bounded / f"{name}-{state}.png",
                                            limits["cases"][name]["hashes"][state]) for state in ("off", "on")}
            donor_boxes = region_boxes(coords, donor, light)
            increments = []
            for role in boxes:
                endpoints = [finish_spatial(target, engine.prepare(target, crop(guide, donor_boxes[role])).apply(target),
                                            STRENGTH) for guide in guides.values()]
                increments.append(endpoints[1].astype(np.int16) - endpoints[0])
            result = composite(full["off"], full["on"], np.stack(increments, axis=-2), weights)
            outside = weights.sum(axis=-1) == 0
            if not np.array_equal(result[outside], full["on"][outside]):
                raise ValueError("Unexpected change outside routed target")
            old[light] = full
            candidate[light] = {"off": full["off"], "on": result}
        cases[name] = dict(scores={"full_reference": score(target, expected, old, areas),
                                   "material_route": score(target, expected, candidate, areas)},
                           outside_fraction=float(outside.mean()),
                           hashes={light: save(out / f"{name}-{light}.png", frames["on"])
                                   for light, frames in candidate.items()},
                           panel_sha=panel(out / f"{name}-comparison.png",
                                           {"target": {"original": target, "desired": expected},
                                            "full reference": {light: f["on"] for light, f in old.items()},
                                            "material route": {light: f["on"] for light, f in candidate.items()}}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=bounded_sha, checkpoint=CHECKPOINT, cases=cases,
             script_sha=digest(script), protocol_sha=digest(script.with_name("gros-material-route-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("bounded_sha")
    run(**vars(parser.parse_args()))
