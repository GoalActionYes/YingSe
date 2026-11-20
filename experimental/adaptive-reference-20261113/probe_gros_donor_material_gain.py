"""Estimate matched-region edit magnitude from the donor's paired RGB response."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_gros_bounded_material_field import support
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import crop, region_boxes
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save


def response(off, on):
    return np.median((on.astype(np.int16) - off).reshape(-1, 3), axis=0)


def estimate(source, current):
    energy = float(current @ current)
    numerator = float(source @ current)
    if min(energy, float(source @ source)) < 1 or numerator <= 0:
        return 1., False
    return float(np.clip(numerator / energy, 0, 4)), True


def apply_gain(off, on, weights, gains):
    total = weights.sum(axis=-1)
    gain = 1 + (weights @ (gains - 1)) / np.maximum(1, total)
    delta = on.astype(np.int16) - off
    return np.rint(np.clip(off + delta * gain[..., None], 0, 255)).astype(np.uint8)


def run(out, parent_sha):
    route = BASE / "gros-material-route-v1"
    bounded = BASE / "gros-bounded-material-field-v3"
    root = BASE / "gros-film-b-cross-v2"
    parent = checked_json(route / "results.json", parent_sha)
    limits = checked_json(bounded / "results.json", parent["parent"])
    film = checked_json(root / "results.json", limits["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, row in parent["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(coords, person, "cold")
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        weights = np.stack([support(target.shape[:2], [bounds]) for bounds in boxes.values()], axis=-1)
        old, new, readings = {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                refs = {key: checked_rgb(root / f"{donor}-cold-{key}.png",
                                            film["inputs"][f"{donor}-cold"]["hashes"][key]) for key in ("off", "on")}
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                refs = {key: checked_rgb(bounded / f"{donor}-bounded-{key}.png",
                                            limits["prepared"][donor]["hashes"][key]) for key in ("off", "on")}
                off = checked_rgb(bounded / f"{name}-off.png", limits["cases"][name]["hashes"]["off"])
            on = checked_rgb(route / f"{name}-{light}.png", row["hashes"][light])
            donor_boxes = region_boxes(coords, donor, light)
            decision = {}
            for role, area in boxes.items():
                src = STRENGTH * response(*(crop(frame, donor_boxes[role]) for frame in refs.values()))
                current = response(crop(off, area), crop(on, area))
                gain, used = estimate(src, current)
                decision[role] = dict(source=src.tolist(), routed=current.tolist(), gain=gain, estimated=used)
            gain = np.asarray([entry["gain"] for entry in decision.values()])
            output = apply_gain(off, on, weights, gain)
            outside = weights.sum(axis=-1) == 0
            if not np.array_equal(output[outside], on[outside]):
                raise ValueError("Changed pixels outside the routed regions")
            old[light], new[light] = {"off": off, "on": on}, {"off": off, "on": output}
            readings[light] = decision
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        outcomes = {key: score(target, expected, frames, regions) for key, frames in (("route", old), ("donor", new))}
        cases[name] = dict(estimates=readings, scores=outcomes,
                          hashes={light: save(out / f"{name}-{light}.png", frames["on"])
                                  for light, frames in new.items()},
                          panel_sha=panel(out / f"{name}-comparison.png", {
                              "target": {"original": target, "desired": expected},
                              "route": {light: frames["on"] for light, frames in old.items()},
                              "donor gain": {light: frames["on"] for light, frames in new.items()}}))
    src = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases, script_sha=digest(src),
             protocol_sha=digest(src.with_name("gros-donor-material-gain-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
