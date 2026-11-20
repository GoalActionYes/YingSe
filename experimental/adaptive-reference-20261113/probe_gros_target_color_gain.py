"""Estimate routed edit gain from donor pixels selected by target color."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from probe_gros_bounded_material_field import support
from probe_gros_donor_material_gain import apply_gain, estimate, response
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import crop, region_boxes
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save


def choose(source, grade, target):
    before = source.reshape(-1, 3)
    values, inverse = np.unique(before, axis=0, return_inverse=True)
    if len(values) < 8:
        return np.zeros(target.shape[:2], dtype=bool), np.empty((0, 3))
    changes = grade.reshape(-1, 3).astype(np.int16) - before
    means = np.zeros((len(values), 3), dtype=float)
    np.add.at(means, inverse, changes)
    means /= np.bincount(inverse)[:, None]
    distances, neighbors = cKDTree(values).query(target.reshape(-1, 3), k=8, workers=1)
    valid = distances[:, -1] <= 12
    return valid.reshape(target.shape[:2]), means[neighbors[valid]].mean(axis=1)


def run(out, base_sha):
    directory = BASE / "gros-donor-material-gain-v1"
    route = BASE / "gros-material-route-v1"
    bounded = BASE / "gros-bounded-material-field-v3"
    root = BASE / "gros-film-b-cross-v2"
    base = checked_json(directory / "results.json", base_sha)
    parent = checked_json(route / "results.json", base["parent"])
    limits = checked_json(bounded / "results.json", parent["parent"])
    film = checked_json(root / "results.json", limits["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, row in base["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(coords, person, "cold")
        areas = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        weights = np.stack([support(target.shape[:2], [box]) for box in boxes.values()], axis=-1)
        baseline, result, decisions = {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                refs = [checked_rgb(root / f"{donor}-cold-{key}.png", film["inputs"][f"{donor}-cold"]["hashes"][key])
                        for key in ("off", "on")]
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                refs = [checked_rgb(bounded / f"{donor}-bounded-{key}.png", limits["prepared"][donor]["hashes"][key])
                        for key in ("off", "on")]
                off = checked_rgb(bounded / f"{name}-off.png", limits["cases"][name]["hashes"]["off"])
            routed = checked_rgb(route / f"{name}-{light}.png", parent["cases"][name]["hashes"][light])
            prior_frame = checked_rgb(directory / f"{name}-{light}.png", row["hashes"][light])
            donor_boxes = region_boxes(coords, donor, light)
            role_rows = {}
            for role, box in boxes.items():
                mask, transfer = choose(*(crop(ref, donor_boxes[role]) for ref in refs), crop(target, box))
                coverage = float(mask.mean())
                prior = row["estimates"][light][role]["gain"]
                gain, used = prior, False
                if coverage >= .75:
                    direction = response(crop(off, box)[mask], crop(routed, box)[mask])
                    predicted, allowed = estimate(STRENGTH * np.median(transfer, axis=0), direction)
                    if allowed:
                        gain, used = predicted, True
                role_rows[role] = dict(gain=gain, prior_gain=prior, coverage=coverage, conditioned=used)
            adjusted = apply_gain(off, routed, weights, np.asarray([v["gain"] for v in role_rows.values()]))
            outside = weights.sum(axis=-1) == 0
            if not np.array_equal(adjusted[outside], prior_frame[outside]):
                raise ValueError("Changed unsupported pixels")
            baseline[light], result[light] = {"off": off, "on": prior_frame}, {"off": off, "on": adjusted}
            decisions[light] = role_rows
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        scores = {key: score(target, expected, r, areas) for key, r in (("prior", baseline), ("conditioned", result))}
        cases[name] = dict(decisions=decisions, scores=scores,
                          hashes={light: save(out / f"{name}-{light}.png", values["on"])
                                  for light, values in result.items()},
                          panel_sha=panel(out / f"{name}-comparison.png", {
                              "target": {"original": target, "desired": expected},
                              "donor gain": {light: s["on"] for light, s in baseline.items()},
                              "target colors": {light: s["on"] for light, s in result.items()}}))
    path = Path(__file__)
    json_to(out / "results.json", dict(parent=base_sha, cases=cases, script_sha=digest(path),
             protocol_sha=digest(path.with_name("gros-target-color-gain-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("base_sha")
    run(**vars(parser.parse_args()))
