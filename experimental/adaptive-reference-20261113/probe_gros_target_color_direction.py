"""Change an existing regional edit direction using matched donor colors."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_gros_bounded_material_field import support
from probe_gros_donor_material_gain import response
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import crop, region_boxes
from probe_gros_target_color_gain import choose
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


def redirect(desired, current):
    norm = float(np.linalg.norm(desired))
    strength = float(np.linalg.norm(current))
    if min(norm, strength) < 1:
        return np.zeros(3), False
    return desired * (strength / norm) - current, True


def combine(image, weights, adjustments):
    offset = weights @ adjustments / np.maximum(weights.sum(axis=-1), 1)[..., None]
    candidate = image + offset
    clipped = float(((candidate > 255) | (candidate < 0)).any(axis=-1).mean())
    return np.rint(np.clip(candidate, 0, 255)).astype(np.uint8), clipped


def run(out, base_sha):
    directory = BASE / "gros-target-color-gain-v1"
    donor = BASE / "gros-donor-material-gain-v1"
    route = BASE / "gros-material-route-v1"
    bounded = BASE / "gros-bounded-material-field-v3"
    root = BASE / "gros-film-b-cross-v2"
    base = checked_json(directory / "results.json", base_sha)
    source = checked_json(donor / "results.json", base["parent"])
    linked = checked_json(route / "results.json", source["parent"])
    locks = checked_json(bounded / "results.json", linked["parent"])
    film = checked_json(root / "results.json", locks["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, row in base["cases"].items():
        person, guide_person = name.split("-from-")
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(coords, person, "cold")
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        weights = np.stack([support(target.shape[:2], [box]) for box in boxes.values()], axis=-1)
        baseline, result, decisions = {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                refs = [checked_rgb(root / f"{guide_person}-cold-{key}.png",
                                   film["inputs"][f"{guide_person}-cold"]["hashes"][key]) for key in ("off", "on")]
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                refs = [checked_rgb(bounded / f"{guide_person}-bounded-{key}.png",
                                   locks["prepared"][guide_person]["hashes"][key]) for key in ("off", "on")]
                off = checked_rgb(bounded / f"{name}-off.png", locks["cases"][name]["hashes"]["off"])
            on = checked_rgb(directory / f"{name}-{light}.png", row["hashes"][light])
            donor_boxes = region_boxes(coords, guide_person, light)
            audit, corrections = {}, []
            for role, box in boxes.items():
                kept, sample = choose(*(crop(a, donor_boxes[role]) for a in refs), crop(target, box))
                coverage = float(kept.mean())
                if coverage != row["decisions"][light][role]["coverage"]:
                    raise ValueError("Changed nearest-color support")
                offset, adjusted = np.zeros(3), False
                if coverage >= .75:
                    current = response(crop(off, box)[kept], crop(on, box)[kept])
                    offset, adjusted = redirect(np.median(sample, axis=0), current)
                corrections.append(offset)
                audit[role] = dict(coverage=coverage, offset=offset.tolist(), adjusted=adjusted)
            changed, clipped = combine(on, weights, np.asarray(corrections))
            outside = weights.sum(axis=-1) == 0
            if not np.array_equal(on[outside], changed[outside]):
                raise ValueError("Changed pixels outside registered regions")
            baseline[light], result[light] = {"off": off, "on": on}, {"off": off, "on": changed}
            decisions[light] = dict(regions=audit, clipped_fraction=clipped)
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        scores = {label: score(target, expected, frames, regions)
                  for label, frames in (("prior", baseline), ("direction", result))}
        cases[name] = dict(scores=scores, decisions=decisions,
                          hashes={light: save(out / f"{name}-{light}.png", image["on"])
                                  for light, image in result.items()},
                          panel_sha=panel(out / f"{name}-comparison.png", {
                              "target": {"original": target, "desired": expected},
                              "prior": {l: f["on"] for l, f in baseline.items()},
                              "direction": {l: f["on"] for l, f in result.items()}}))
    src = Path(__file__)
    json_to(out / "results.json", dict(parent=base_sha, cases=cases, script_sha=digest(src),
             protocol_sha=digest(src.with_name("gros-target-color-direction-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("base_sha")
    run(**vars(parser.parse_args()))
