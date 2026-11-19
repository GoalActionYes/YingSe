"""Measure whether donor diversity improves a known Film B color operator."""

import argparse
from pathlib import Path

import numpy as np

from probe_aligned_cubic_pilot import POWERS, RIDGE, cubic_mapping, design, thumbnail_pair
from probe_arri_film_b_dose import endpoints
from probe_bg_film_b_reference_path import SOURCE, SOURCE_SHA
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_frozen_candidate_development_crosscheck import BASE, STRENGTH
from probe_frozen_render_blend_bound import neutral
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_paired_cubic_cross_content import apply
from probe_target_mirror_chroma import checked_rgb, panel, save
from reference_color_app.lut import apply_mapping
from verify_development_inputs import manifest

ROLES = ("C1", "F1")


def prepare(donor):
    a, b = thumbnail_pair(*donor)
    yy, xx = np.indices(a.shape[:2])
    train = (xx // 8 + yy // 8) % 2 == 0
    return a, b, train


def learn(donors):
    nonlinear = np.diag([sum(power) >= 2 for power in POWERS]).astype(float)
    prepared = [(design(a[train].astype(float) / 255), (b[train].astype(float) - a[train]) / 255,
                 b[train]) for a, b, train in donors]
    channels = []
    for channel in range(3):
        blocks, values = [], []
        for x, delta, b in prepared:
            usable = (b[:, channel] > 2) & (b[:, channel] < 253)
            count = int(usable.sum())
            if count < 128:
                raise ValueError("Insufficient donor support")
            normalization = np.sqrt(count * len(prepared))
            blocks.append(x[usable] / normalization)
            values.append(delta[usable, channel] / normalization)
        x = np.vstack([*blocks, np.sqrt(RIDGE) * nonlinear])
        y = np.concatenate([*values, np.zeros(len(POWERS))])
        weights, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != len(POWERS) or not np.isfinite(weights).all():
            raise ValueError("Unstable grouped fit")
        channels.append(weights)
    return cubic_mapping(np.stack(channels, axis=1))


def donor_fit():
    prep = manifest(BASE, "arri-two-looks-v2")
    folder = BASE / "arri-two-looks-v2"
    donors = {name: prepare(endpoints(folder, prep, name)) for name in prep["sources"]}
    subsets = {"pooled": tuple(donors)}
    subsets.update({f"without-{held}": tuple(name for name in donors if name != held) for held in donors})
    tables = {key: learn([donors[name] for name in names]) for key, names in subsets.items()}
    validation = {}
    for key, table in tables.items():
        validation[key] = {}
        for name, (a, b, train) in donors.items():
            error = np.abs(apply_mapping(a, table, 1).astype(np.int16) - b)
            validation[key][name] = {"in_fit": name in subsets[key], "mae": float(error[~train].mean()),
                                     "p95": float(np.quantile(error[~train], .95))}
    return tables, validation


def errors(render, known, regions):
    delta = np.abs(render.astype(np.int16) - known)
    return {name: float(delta[mask].mean()) for name, mask in regions.items()}


def render_targets(out, tables, source):
    prior = source["comparison"]["paired"]["cases"]
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    rows = {}
    for name, archive in manifest(BASE, "bg-real-editor-look")["cases"].items():
        target = checked_rgb(BASE / "bg-real-editor-look" / name / "target.png", archive["artifacts"]["target.png"])
        known = checked_rgb(BASE / SOURCE / f"{name}-expected.png", source["qualification"][name]["expected_sha256"])
        person = mask_for(entries[name], name.split("-")[0])
        neutral_area = neutral(target)
        scores = evaluate(target, {"off": target, "on": known},
                          dict.fromkeys(ROLES, {"off": target, "on": known}), person)
        if not scores["passed"]:
            raise ValueError("Changed development qualification")
        areas = {"whole": np.ones(person.shape, dtype=bool), "person": person, "neutral": neutral_area}
        old = {role: checked_rgb(BASE / SOURCE / f"paired-{name}-{role}-on.png", prior[name]["outputs"][f"{role}-on"])
               for role in ROLES}
        baseline = {role: errors(result, known, areas) for role, result in old.items()}
        results = {key: apply(target, table) for key, table in tables.items()}
        record = {}
        for role, output in results.items():
            stats = evaluate(target, {"off": target, "on": known},
                             dict.fromkeys(ROLES, {"off": target, "on": output}), person)
            zones = errors(output, known, areas)
            record[role] = {"absolute_mae": zones, "projection": stats["projection"]["C1"],
                            "passed": stats["passed"] and max(zones.values()) <= 3,
                            "sha256": save(out / f"{name}-{role}.png", output)}
        other_fits = [value[neutral_area].astype(np.int16) for key, value in results.items() if key != "pooled"]
        max_change = max(float(np.abs(x - y).mean()) for i, x in enumerate(other_fits) for y in other_fits[i + 1:])
        board = {"known style": {"target": target, "single C1": old["C1"], "single F1": old["F1"],
                                 "pooled": results["pooled"], "expected": known}}
        rows[name] = {"single_reference_mae": baseline, "candidate": record, "leave_one_neutral_max_change": max_change,
                      "panel_sha256": panel(out / f"{name}-comparison.png", board)}
    return rows


def run(out):
    source = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    if source["qualified_cases"] != 4 or source["strength"] != STRENGTH:
        raise ValueError("Original development test changed")
    tables, diagnostics = donor_fit()
    out.mkdir()
    rows = render_targets(out, tables, source)
    count = sum(row["candidate"]["pooled"]["passed"] for row in rows.values())
    worst = float(max(row["candidate"]["pooled"]["absolute_mae"][region]
                      - np.mean([scores[region] for scores in row["single_reference_mae"].values()])
                      for row in rows.values() for region in ("person", "neutral")))
    json_to(out / "results.json", {
        "source_sha256": SOURCE_SHA, "strength": STRENGTH, "donor_checks": diagnostics, "cases": rows,
        "passing_cases": count, "worst_protected_regression": worst, "passed": count >= 3 and worst <= 2,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    run(cli.parse_args().out)
