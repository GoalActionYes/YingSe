"""Offline bound for changing default colors while preserving Film B response."""
import argparse
from pathlib import Path

import numpy as np

from probe_bg_film_b_reference_path import (
    BASE, PARENT, PARENT_SHA, SOURCE, SOURCE_SHA, checked_rgb, evaluate,
    json_to, manifest, mask_for, panel, save, verified_json,
)
from background_intervention import target_regions

ROLES = ("C1", "F1")
DONOR = "film-b-donor-appearance-v2"
DONOR_EVIDENCE = "docs/research/2026-11-19-film-b-donor-appearance-evidence.json"
ROOT = Path(__file__).resolve().parents[2]


def response_balanced(target, rgb):
    d, e = [rgb[k]["on"].astype(np.int16) - rgb[k]["off"] for k in ROLES]
    x = -np.rint((d - e) * .5).astype(np.int16)
    lo = np.maximum(np.maximum(0, -d), np.maximum(0, -e) + x)
    hi = np.minimum(np.minimum(255, 255 - d), np.minimum(255, 255 - e) + x)
    if np.any(lo > hi):
        raise ValueError("No feasible RGB8 baseline")
    a = np.clip(np.rint(target + x * .5).astype(np.int16), lo, hi)
    b = a - x
    result = dict(zip(ROLES, ({"off": a, "on": a + d}, {"off": b, "on": b + e}), strict=True))
    for vals, expected in zip(result.values(), (d, e), strict=True):
        for key in vals:
            assert vals[key].min() >= 0 and vals[key].max() <= 255
            vals[key] = vals[key].astype(np.uint8)
        np.testing.assert_array_equal(vals["on"].astype(np.int16) - vals["off"], expected)
    return result, np.abs(d - e) * .5


def run(out):
    import json
    source = BASE / SOURCE
    base = verified_json(source / "results.json", SOURCE_SHA)
    receipt = json.loads((ROOT / DONOR_EVIDENCE).read_bytes())
    donor = verified_json(BASE / DONOR / "results.json", receipt.pop("report_sha256"))
    if donor != receipt or base["qualified_cases"] != 4:
        raise ValueError("Frozen source changed")
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    cases = {}
    out.mkdir()
    for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        target = checked_rgb(BASE / "bg-real-editor-look" / key / "target.png", row["artifacts"]["target.png"])
        author = checked_rgb(source / f"{key}-expected.png", base["qualification"][key]["expected_sha256"])
        truth = {"off": target, "on": author}
        person = mask_for(entries[key], key.split("-")[0])
        neutral = target_regions(target, person)["neutral_proxy"]
        areas = dict(person=person, neutral=neutral, neutral_person=neutral & person)
        entry = base["comparison"]["expanded"]["cases"][key]
        native = {role: {s: checked_rgb(source / f"expanded-{key}-{role}-{s}.png", entry["outputs"][f"{role}-{s}"])
                         for s in ("off", "on")} for role in ROLES}
        received = {role: {"off": native[role]["off"],
                           "on": checked_rgb(BASE / DONOR / f"{key}-{role}-replay.png",
                                             donor["cases"][key]["outputs"][role])} for role in ROLES}
        cases[key] = {}
        for arm, rgb, old in (("expanded", native, entry["scores"]),
                              ("replay", received, donor["cases"][key]["scores"]["replay"])):
            assert evaluate(target, truth, rgb, person) == old
            balanced, lower = response_balanced(target, rgb)
            observed = evaluate(target, truth, balanced, person)
            assert observed["projection"] == old["projection"]
            bounds = {k: float(lower[m].mean()) for k, m in areas.items() if m.sum() >= 256}
            for k, floor in bounds.items():
                observed_max = max(vals[k] for vals in observed["swap_mae"].values())
                assert floor - 1e-12 <= observed_max <= floor + .5 + 1e-12
            names = {f"{k}-{state}": save(out / f"{key}-{arm}-{k}-{state}.png", data)
                     for k, states in balanced.items() for state, data in states.items()}
            board = {f"{k} {s}": {"target": target, "expected": truth[s], "frozen": rgb[k][s], "balanced": data}
                     for k, states in balanced.items() for s, data in states.items()}
            cases[key][arm] = {
                "original": old, "balanced": observed, "best_possible_lower_bound": bounds, "outputs": names,
                "comparison_sha256": panel(out / f"{key}-{arm}-comparison.png", board),
            }
    json_to(out / "results.json", {"cases": cases, "source_sha256": SOURCE_SHA,
                                   "donor_weight_sha256": donor["weight_sha256"]})
    for key, rows in cases.items():
        print(key, {k: (max(v["best_possible_lower_bound"].values()), v["balanced"]["max_protected_swap"],
                        v["balanced"]["passed"]) for k, v in rows.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
