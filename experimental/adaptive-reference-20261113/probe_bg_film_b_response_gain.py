"""Test whether the measured Film B response needs only two shared gains."""

import argparse
from pathlib import Path

import numpy as np

from probe_bg_film_b_reference_path import SOURCE, SOURCE_SHA
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for, split_signal
from probe_frozen_candidate_development_crosscheck import BASE, COUNTS, STRENGTH
from probe_frozen_render_blend_bound import Y
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb, panel, save
from verify_development_inputs import manifest, verify

ROLES = ("C1", "F1")


def fit_gain(measurements, expected):
    power = np.square(expected).sum()
    if power <= 0:
        raise ValueError("Expected edit is not measurable")
    cross = np.array([(component * expected).sum() for component in measurements])
    energy = sum(np.square(component).sum() for component in measurements)
    free = max(0., float(cross.sum() / energy)) if energy > 0 else 0.
    slopes = cross / power
    interval = None
    if np.all(slopes > 0):
        low, high = float(max(.75 / slopes)), float(min(1.25 / slopes))
        if low <= high:
            interval = (low, high)
    return {"native_projections": slopes.tolist(), "common_interval": interval, "free_gain": free,
            "selected_gain": free if interval is None else float(np.clip(free, *interval))}


def grade(target, delta, gains):
    y = (delta @ Y)[..., None]
    color = delta - y
    rgb = target + gains["luma"]["selected_gain"] * y + gains["chroma"]["selected_gain"] * color
    clipped = float(((rgb < 0) | (rgb > 255)).mean())
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), clipped


def probe(out, prior, entries):
    records = {}
    for key, item in manifest(BASE, "bg-real-editor-look")["cases"].items():
        original = checked_rgb(BASE / "bg-real-editor-look" / key / "target.png", item["artifacts"]["target.png"])
        person = mask_for(entries[key], key.split("-")[0])
        expected = checked_rgb(BASE / SOURCE / f"{key}-expected.png", prior["qualification"][key]["expected_sha256"])
        wanted = split_signal(expected, original)
        rows = {}
        for model in ("13", "expanded"):
            archived = prior["comparison"][model]["cases"][key]
            values = {role: {state: checked_rgb(BASE / SOURCE / f"{model}-{key}-{role}-{state}.png",
                                               archived["outputs"][f"{role}-{state}"])
                             for state in ("off", "on")} for role in ROLES}
            measured = {role: split_signal(states["on"], states["off"]) for role, states in values.items()}
            gains = {name: fit_gain([measured[role][name] for role in ROLES], known) for name, known in wanted.items()}
            variants, diagnostics, boards = {}, {}, {}
            for role, states in values.items():
                result, clipped = grade(original, states["on"].astype(np.int16) - states["off"], gains)
                variants[role] = {"off": original, "on": result}
                diagnostics[role] = {"clipped_channels": clipped, "sha256": save(out / f"{model}-{key}-{role}.png", result)}
                boards[role] = {"target": original, "ordinary 2-image": states["on"],
                                "calibrated response": result, "known edit": expected}
            scores = evaluate(original, {"off": original, "on": expected}, variants, person)
            rows[model] = {"gains": gains, "candidate": scores, "archived": archived["scores"], "outputs": diagnostics,
                           "panel_sha256": panel(out / f"{model}-{key}-comparison.png", boards)}
        records[key] = rows
    return records


def run(out):
    verify(BASE, COUNTS)
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    if prior["strength"] != STRENGTH or prior["qualified_cases"] != 4:
        raise ValueError("Frozen test qualification changed")
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    out.mkdir()
    records = probe(out, prior, entries)
    summary = {}
    for arm in ("13", "expanded"):
        rows = [row[arm] for row in records.values()]
        count = sum(row["candidate"]["passed"] for row in rows)
        worst = max(row["candidate"]["max_protected_swap"] for row in rows)
        summary[arm] = {"passing_cases": count, "max_protected_swap": worst, "passed": count >= 3 and worst <= 3}
    json_to(out / "results.json", {"parent_sha256": SOURCE_SHA, "strength": STRENGTH, "cases": records, "summary": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
