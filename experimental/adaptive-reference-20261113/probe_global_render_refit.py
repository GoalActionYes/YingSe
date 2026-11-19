"""Assess a position-independent refit of frozen two-image RGB8 outputs."""
import argparse
from pathlib import Path

import numpy as np

from probe_aligned_cubic_pilot import cubic_mapping, learn, thumbnail_pair
from probe_bg_film_b_reference_path import BASE, PARENT, PARENT_SHA, SOURCE, SOURCE_SHA, evaluate, mask_for
from probe_facd_two_image_baseline import STYLES, case_metrics, read_checked
from probe_frozen_render_blend_bound import FULL, FULL_SHA
from probe_hdm_hdr_display import json_to
from probe_target_mirror_chroma import (
    FACD_SHA, FACD_SOURCE_SHA, ROOT, checked_json, checked_rgb, interior, panel, save, target_regions,
)
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial
from verify_development_inputs import manifest


def refit(target, teacher):
    a, b = thumbnail_pair(target, teacher)
    y, x = np.indices(a.shape[:2])
    train = (x // 8 + y // 8) % 2 == 0
    weights = learn(a, b, train.ravel())
    if weights is None:
        return teacher, {"fallback": True}
    mapping = cubic_mapping(weights)
    error = np.abs(apply_mapping(a, mapping, 1).astype(np.int16) - b)[~train]
    return apply_mapping(target, mapping, 1), {
        "fallback": False, "holdout_mae": float(error.mean()), "holdout_p95": float(np.quantile(error, .95)),
    }


def film(out):
    source = BASE / SOURCE
    prior = checked_json(source / "results.json", SOURCE_SHA)
    entries = {row["pair"]: row for row in checked_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    cases = {}
    for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        target = checked_rgb(BASE / "bg-real-editor-look" / key / "target.png", row["artifacts"]["target.png"])
        frozen = prior["comparison"]["expanded"]["cases"][key]
        raw, projected, details = {}, {}, {}
        for role in ("C1", "F1"):
            raw[role], projected[role] = {}, {}
            for state in ("off", "on"):
                tag = f"{role}-{state}"
                raw[role][state] = checked_rgb(source / f"expanded-{key}-{tag}.png", frozen["outputs"][tag])
                projected[role][state], details[tag] = refit(target, raw[role][state])
                details[tag]["sha256"] = save(out / f"{key}-{tag}.png", projected[role][state])
        expected = checked_rgb(source / f"{key}-expected.png", prior["qualification"][key]["expected_sha256"])
        person = mask_for(entries[key], key.split("-")[0])
        scores = [evaluate(target, {"off": target, "on": expected}, arm, person) for arm in (raw, projected)]
        if scores[0] != frozen["scores"]:
            raise ValueError(f"Film B baseline changed: {key}")
        comparison = panel(out / f"{key}-comparison.png", {
            role: {"target": target, "known edit": expected, "expanded": raw[role]["on"], "refit": projected[role]["on"]}
            for role in raw
        })
        cases[key] = {"expanded": scores[0], "refit": scores[1], "outputs": details, "comparison_sha256": comparison}
    count = sum(row["refit"]["passed"] for row in cases.values())
    worst = max(row["refit"]["max_protected_swap"] for row in cases.values())
    return {"passing_cases": count, "worst_protected_swap": worst, "passed": count >= 3 and worst <= 3, "cases": cases}


def facd(out):
    expanded = checked_json(BASE / FULL / "results.json", FULL_SHA)["suites"]["facd"]
    baseline = checked_json(BASE / "facd-two-image-baseline-v1" / "results.json", FACD_SHA)
    source = checked_json(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json", FACD_SOURCE_SHA)
    aligned = BASE / "facd-border-registration-v1"
    cases = {}
    for group, refs in baseline["references"].items():
        root = out / group
        root.mkdir()
        target = read_checked(aligned, group, "Origin", source)
        frozen = expanded["cases"][group]
        variants, detail = {style: {} for style in STYLES}, {}
        for style in STYLES:
            for ref in refs:
                name = f"{ref}-{style}.png"
                record = frozen["outputs"][name]
                teacher = checked_rgb(BASE / FULL / "facd" / group / name, record["sha256"])
                result, detail[name] = refit(target, teacher)
                variants[style][ref] = result.astype(np.int16)
                detail[name]["sha256"] = save(root / name, result)
        endpoints = {style: read_checked(aligned, group, style, source) for style in STYLES}
        region = target_regions(target, np.zeros(target.shape[:2], dtype=bool))["neutral_proxy"]
        region &= np.logical_and.reduce([interior(image) for image in (target, *endpoints.values())])
        if int(region.sum()) != frozen["common_neutral_pixels"]:
            raise ValueError(f"FACD neutral region changed: {group}")
        expected = {style: finish_spatial(target, grade, .8).astype(np.int16) for style, grade in endpoints.items()}
        for style in STYLES:
            for ref in refs:
                name = f"{ref}-{style}.png"
                error = float(np.abs(variants[style][ref][region] - expected[style][region]).mean())
                detail[name]["neutral_error"] = error
                detail[name]["neutral_extra_vs_13"] = error - frozen["outputs"][name]["neutral_error"]["baseline"]
        cases[group] = {"refit": case_metrics(target, expected, variants, refs), "expanded": frozen["candidate"],
                        "outputs": detail}
    count = sum(all(row["refit"]["gates"].values()) for row in cases.values())
    worst = max(image["neutral_extra_vs_13"] for row in cases.values() for image in row["outputs"].values())
    return {"passing_cases": count, "worst_neutral_extra": worst, "passed": count >= 6 and worst <= 2,
            "expanded_passing_cases": expanded["passing_cases"], "cases": cases}


def run(out):
    out.mkdir()
    for group in ("film", "facd"):
        (out / group).mkdir()
    results = {"film": film(out / "film"), "facd": facd(out / "facd")}
    json_to(out / "results.json", results)
    print({k: {a: b for a, b in v.items() if a != "cases"} for k, v in results.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
