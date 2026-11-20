"""Calibrate regional edit magnitude after correcting its direction."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_gros_bounded_material_field import support
from probe_gros_donor_material_gain import response
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import crop, region_boxes
from probe_gros_target_color_direction import combine
from probe_gros_target_color_gain import choose
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save


def rescale(sample, current):
    source = np.linalg.norm(np.median(sample, axis=0))
    initial = np.linalg.norm(current)
    if min(source, initial) < 1:
        return np.zeros(3), 1.
    gain = float(np.clip(STRENGTH * source / initial, 0, 4))
    return (gain - 1) * current, gain


def run(out, parent_sha):
    directory = BASE / "gros-target-color-direction-v1"
    parent = checked_json(directory / "results.json", parent_sha)
    previous = checked_json(BASE / "gros-target-color-gain-v1/results.json", parent["parent"])
    donor = checked_json(BASE / "gros-donor-material-gain-v1/results.json", previous["parent"])
    route = checked_json(BASE / "gros-material-route-v1/results.json", donor["parent"])
    bounded = BASE / "gros-bounded-material-field-v3"
    correction = checked_json(bounded / "results.json", route["parent"])
    root = BASE / "gros-film-b-cross-v2"
    film = checked_json(root / "results.json", correction["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, row in parent["cases"].items():
        person, reference = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(coords, person, "cold")
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        weights = np.stack([support(target.shape[:2], [box]) for box in boxes.values()], axis=-1)
        baseline, result, decisions = {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                refs = [checked_rgb(root / f"{reference}-cold-{key}.png",
                                    film["inputs"][f"{reference}-cold"]["hashes"][key]) for key in ("off", "on")]
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                refs = [checked_rgb(bounded / f"{reference}-bounded-{key}.png",
                                    correction["prepared"][reference]["hashes"][key]) for key in ("off", "on")]
                off = checked_rgb(bounded / f"{name}-off.png", correction["cases"][name]["hashes"]["off"])
            on = checked_rgb(directory / f"{name}-{light}.png", row["hashes"][light])
            source_boxes = region_boxes(coords, reference, light)
            audit, changes = {}, []
            for role, box in boxes.items():
                keep, sample = choose(*(crop(ref, source_boxes[role]) for ref in refs), crop(target, box))
                coverage = float(keep.mean())
                if coverage != row["decisions"][light]["regions"][role]["coverage"]:
                    raise ValueError("Changed nearby colors")
                change, gain = np.zeros(3), 1.
                if coverage >= .75:
                    change, gain = rescale(sample, response(crop(off, box)[keep], crop(on, box)[keep]))
                changes.append(change)
                audit[role] = dict(gain=gain, coverage=coverage)
            changed, clipped = combine(on, weights, np.asarray(changes))
            outside = weights.sum(axis=-1) == 0
            if not np.array_equal(changed[outside], on[outside]):
                raise ValueError("Changed pixels outside regions")
            baseline[light], result[light] = {"off": off, "on": on}, {"off": off, "on": changed}
            decisions[light] = dict(regions=audit, clipped_fraction=clipped)
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        scores = {key: score(target, expected, values, regions)
                  for key, values in (("direction", baseline), ("strength", result))}
        cases[name] = dict(scores=scores, decisions=decisions,
                          hashes={light: save(out / f"{name}-{light}.png", frames["on"])
                                  for light, frames in result.items()},
                          panel_sha=panel(out / f"{name}-comparison.png", {
                              "target": {"original": target, "desired": expected},
                              "direction": {l: f["on"] for l, f in baseline.items()},
                              "strength": {l: f["on"] for l, f in result.items()}}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases, script_sha=digest(script),
             protocol_sha=digest(script.with_name("gros-directed-chroma-strength-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
