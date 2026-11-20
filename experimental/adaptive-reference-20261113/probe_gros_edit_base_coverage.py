"""Separate target base drift from the area that receives the intended edit."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_gros_bounded_material_field import support
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import region_boxes
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


def replay(target, off, on):
    edit = on.astype(np.int16) - off
    combined = target.astype(np.int16) + edit
    clipping = (combined.min(axis=-1) < 0) | (combined.max(axis=-1) > 255)
    return np.clip(combined, 0, 255).astype(np.uint8), float(clipping.mean())


def run(out, parent_sha):
    source = BASE / "gros-directed-chroma-strength-v1"
    bounded = BASE / "gros-bounded-material-field-v3"
    root = BASE / "gros-film-b-cross-v2"
    current = checked_json(source / "results.json", parent_sha)
    direction = checked_json(BASE / "gros-target-color-direction-v1/results.json", current["parent"])
    gains = checked_json(BASE / "gros-target-color-gain-v1/results.json", direction["parent"])
    donor = checked_json(BASE / "gros-donor-material-gain-v1/results.json", gains["parent"])
    route = checked_json(BASE / "gros-material-route-v1/results.json", donor["parent"])
    fixed = checked_json(bounded / "results.json", route["parent"])
    film = checked_json(root / "results.json", fixed["film"])
    portrait = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != portrait["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, row in current["cases"].items():
        person = name.split("-from-")[0]
        rgb = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        boxes = region_boxes(coords, person, "cold")
        domain = support(rgb.shape[:2], boxes.values()) > 0
        regions = masks(coords[person]["cold"], coords["screen"], rgb.shape[:2])
        original, neutral, inside, checks = {}, {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                off = checked_rgb(bounded / f"{name}-off.png", fixed["cases"][name]["hashes"]["off"])
            on = checked_rgb(source / f"{name}-{light}.png", row["hashes"][light])
            result, clipping = replay(rgb, off, on)
            original[light] = {"off": off, "on": on}
            neutral[light] = {"off": rgb, "on": result}
            checks[light] = dict(clipping=clipping)
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        for light, frames in neutral.items():
            now = frames["on"]
            completed = np.where(domain[..., None], expected, now)
            before = np.abs(original[light]["on"].astype(np.int16) - expected)
            error = np.abs(now.astype(np.int16) - expected)
            lower = float(error[~domain].sum() / error.size)
            achieved = float(np.abs(completed.astype(np.int16) - expected).mean())
            if not np.isclose(achieved, lower, atol=1e-12, rtol=0):
                raise ValueError("Support-constrained MAE bound did not match")
            inside[light] = {"off": rgb, "on": completed}
            checks[light].update(minimum_with_current_support=lower, original_on_mae=float(before.mean()),
                                 neutral_on_mae=float(error.mean()), outside_mae=float(error[~domain].mean()))
        arms = {"original": original, "target_base": neutral, "ideal_inside": inside}
        cases[name] = dict(support_fraction=float(domain.mean()), checks=checks,
                           scores={key: score(rgb, expected, values, regions) for key, values in arms.items()},
                           hashes={f"{arm}-{light}": save(out / f"{name}-{arm}-{light}.png", frames["on"])
                                   for arm in ("target_base", "ideal_inside") for light, frames in arms[arm].items()},
                           panel_sha=panel(out / f"{name}-comparison.png",
                                           {"target": {"original": rgb, "desired": expected}} |
                                           {arm: {l: f["on"] for l, f in v.items()} for arm, v in arms.items()}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, cases=cases, script_sha=digest(script),
             protocol_sha=digest(script.with_name("gros-edit-base-coverage-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
