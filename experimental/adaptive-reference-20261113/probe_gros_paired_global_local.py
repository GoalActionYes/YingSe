"""Test full-image paired color replay with and without preserving local improvements."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_gros_bounded_material_field import support
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import region_boxes
from probe_paired_cubic_cross_content import paired_fit
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save
from reference_color_app.lut import apply_mapping


def blend(global_frame, local_frame, weight):
    different = local_frame.astype(np.int16) - global_frame
    return np.rint(global_frame + different * weight[..., None]).astype(np.uint8)


def chain(root, parent):
    current = parent
    for name in ("gros-directed-chroma-strength-v1", "gros-target-color-direction-v1",
                 "gros-target-color-gain-v1", "gros-donor-material-gain-v1",
                 "gros-material-route-v1", "gros-bounded-material-field-v3"):
        current = checked_json(root / name / "results.json", current["parent"])
    return current


def run(out, parent_sha):
    directory = BASE / "gros-edit-base-coverage-v1"
    parent = checked_json(directory / "results.json", parent_sha)
    limits = chain(BASE, parent)
    film_root = BASE / "gros-film-b-cross-v2"
    bounded = BASE / "gros-bounded-material-field-v3"
    film = checked_json(film_root / "results.json", limits["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    labels = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(labels) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    zones = json.loads(labels.read_bytes())
    tables = {}
    for person, row in limits["prepared"].items():
        cold = [checked_rgb(film_root / f"{person}-cold-{state}.png",
                            film["inputs"][f"{person}-cold"]["hashes"][state]) for state in ("off", "on")]
        warm = [checked_rgb(bounded / f"{person}-bounded-{state}.png",
                            row["hashes"][state]) for state in ("off", "on")]
        tables[person] = {"cold": paired_fit(*cold), "warm": paired_fit(*warm)}
    out.mkdir()
    cases = {}
    for name, previous in parent["cases"].items():
        person, donor = name.split("-from-")
        target = checked_rgb(film_root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(zones, person, "cold")
        local = support(target.shape[:2], boxes.values())
        regions = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
        arms = {label: {} for label in ("prior", "global", "hybrid")}
        for light, (table, _) in tables[donor].items():
            key = f"target_base-{light}"
            old = checked_rgb(directory / f"{name}-{key}.png", previous["hashes"][key])
            generated = finish_spatial(target, apply_mapping(target, table, 1), STRENGTH)
            combined = blend(generated, old, local)
            if not np.array_equal(combined[local == 1], old[local == 1]):
                raise ValueError("Lost exact local core")
            if not np.array_equal(combined[local == 0], generated[local == 0]):
                raise ValueError("Lost exact global outside")
            for role, image in (("prior", old), ("global", generated), ("hybrid", combined)):
                arms[role][light] = {"off": target, "on": image}
        expected = checked_rgb(film_root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        results = {key: score(target, expected, frames, regions) for key, frames in arms.items()}
        cases[name] = dict(scores=results,
                          hashes={f"{role}-{light}": save(out / f"{name}-{role}-{light}.png", frames["on"])
                                  for role in ("global", "hybrid") for light, frames in arms[role].items()},
                          panel=panel(out / f"{name}-comparison.png", {"target": {"source": target, "desired": expected}} |
                                      {key: {light: f["on"] for light, f in v.items()} for key, v in arms.items()}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases,
        fits={person: {light: pair[1] for light, pair in views.items()} for person, views in tables.items()},
        script_sha=digest(script), protocol_sha=digest(script.with_name("gros-paired-global-local-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
