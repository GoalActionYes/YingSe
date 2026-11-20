"""Measure whether bounded RGB gain suffices on the routed Film B response."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_beyond_edit_response_envelope import projection_bounds
from probe_gros_bounded_material_field import support
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_gros_material_route import region_boxes
from probe_mid_edit_response_envelope import solution
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save


def endpoints(off, on, active):
    edited = on.astype(float)
    start = np.where(active[..., None], off, edited)
    end = np.where(active[..., None], start + 4 * (edited - start), edited)
    return start, end


def run(out, route_sha):
    route = BASE / "gros-material-route-v1"
    root = BASE / "gros-film-b-cross-v2"
    bounded = BASE / "gros-bounded-material-field-v3"
    prior = checked_json(route / "results.json", route_sha)
    correction = checked_json(bounded / "results.json", prior["parent"])
    film = checked_json(root / "results.json", correction["film"])
    original = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    ledger = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(ledger) != original["regions_sha256"]:
        raise ValueError("Changed regions")
    coords = json.loads(ledger.read_bytes())
    out.mkdir()
    cases = {}
    for name, table in prior["cases"].items():
        person = name.split("-from-")[0]
        target = checked_rgb(root / f"{person}-cold-off.png", film["inputs"][f"{person}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{name}-expected.png", film["cases"][name]["expected_sha"])
        boxes = region_boxes(coords, person, "cold")
        active = support(target.shape[:2], boxes.values()) > 0
        regions = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        archived, candidate, audits = {}, {}, {}
        for light in ("cold", "warm"):
            if light == "cold":
                off = checked_rgb(root / f"{name}-13-cold-off.png", film["cases"][name]["hashes"]["13-cold-off"])
            else:
                off = checked_rgb(bounded / f"{name}-off.png", correction["cases"][name]["hashes"]["off"])
            on = checked_rgb(route / f"{name}-{light}.png", table["hashes"][light])
            left, right = endpoints(off, on, active)
            desired = expected.astype(float) + off.astype(float) - target
            continuous, weight = solution(left, right, desired)
            rendered = np.rint(continuous).astype(np.uint8)
            if not np.array_equal(on[~active], rendered[~active]):
                raise ValueError("Changed output outside routing support")
            audits[light] = {}
            for key, mask in regions.items():
                minimum = max(0., float(np.abs(continuous[mask] - desired[mask]).mean()) - .5)
                interval = projection_bounds(left[mask], right[mask], off[mask].astype(float), desired[mask])
                audits[light][key] = dict(mae_floor=minimum, projection_interval=interval,
                    exceeds=minimum > 3 + 1e-10 or interval[1] < .75 - 1e-10 or interval[0] > 1.25 + 1e-10,
                    gain_quartiles=np.quantile(4 * weight[mask & active], [.25,.5,.75]).tolist())
            archived[light] = {"off": off, "on": on}
            candidate[light] = {"off": off, "on": rendered}
        scores = {"original": score(target, expected, archived, regions), "gain": score(target, expected, candidate, regions)}
        for light, details in audits.items():
            for region, row in details.items():
                measured = scores["gain"][region]["response"][light]
                low, high = row["projection_interval"]
                if measured["response_mae"] < row["mae_floor"] - 1e-10 or not low - 1e-10 <= measured["projection"] <= high + 1e-10:
                    raise ValueError("Actual output falls outside analytic bound")
        cases[name] = dict(scores=scores, audit=audits,
                          hashes={light: save(out / f"{name}-{light}.png", frames["on"])
                                  for light, frames in candidate.items()},
                          panel=panel(out / f"{name}-comparison.png",
                                      {"target": {"source": target, "known edit": expected},
                                       "route": {l: f["on"] for l, f in archived.items()},
                                       "gain": {l: f["on"] for l, f in candidate.items()}}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=route_sha, cases=cases, script_sha=digest(script),
            protocol_sha=digest(script.with_name("gros-material-route-gain-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("route_sha")
    run(**vars(parser.parse_args()))
