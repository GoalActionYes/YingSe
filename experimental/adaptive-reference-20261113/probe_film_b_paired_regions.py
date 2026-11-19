"""Does an aligned before/after editing difference transfer better by region?"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from probe_aligned_cubic_pilot import cubic_mapping, learn, thumbnail_pair
from probe_film_b_regional_distribution import (
    BASE, HELD_SHA, PARENT, PARENT_SHA, ROOT, SOURCE, SOURCE_SHA, checked_rgb, evaluate, finish_spatial,
    inside, json_to, manifest, mask_for, panel, save, verified_json,
)
from probe_hdm_hdr_display import digest
from probe_paired_cubic_cross_content import paired_fit, apply
from background_intervention import target_regions
from reference_color_app.lut import apply_mapping


def mappings(before, after, person):
    a, b = thumbnail_pair(before, after)
    h, w = a.shape[:2]
    shape = np.asarray(Image.fromarray(person).resize((w, h), Image.Resampling.NEAREST))
    yy, xx = np.indices((h, w))
    checker = (xx // 8 + yy // 8) % 2 == 0
    tables, support = [], []
    for area in (shape, ~shape):
        train = checker & inside(area)
        support.append(int(train.sum()))
        valid = ((b > 2) & (b < 253)) & train[..., None]
        weights = learn(a, b, train.ravel()) if valid.sum(axis=(0, 1)).min() >= 128 else None
        tables.append(cubic_mapping(weights) if weights is not None else None)
    return tables, support


def run(out):
    source = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    held = verified_json(ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json", HELD_SHA)
    refs = {row["target_pair"]: row["input"] for row in held["cases"].values()}
    parent = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    assert source["qualified_cases"] == 4
    out.mkdir()
    cases = {}
    for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        root = BASE / "bg-real-editor-look" / key
        target = checked_rgb(root / "target.png", row["artifacts"]["target.png"])
        expected = checked_rgb(BASE / SOURCE / f"{key}-expected.png", source["qualification"][key]["expected_sha256"])
        person = mask_for(parent[key], key.split("-")[0])
        regions = target_regions(target, person)
        areas = {"person": person, "neutral": regions["neutral_proxy"], "whole": regions["whole"]}
        opacity = np.asarray(Image.fromarray(person.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(2)),
                             dtype=np.float64)[..., None] / 255
        mask_entry = refs[key]["mask"]
        path = BASE / "bg-real-pixel-inputs" / mask_entry["name"]
        if digest(path) != mask_entry["sha256"]:
            raise ValueError("Reference mask changed")
        with Image.open(path) as image:
            ref_person = np.asarray(image.convert("L")) >= 128
        rendered = {arm: {} for arm in ("global", "regional")}
        rows, receipts = {}, {}
        for role in ("C1", "F1"):
            name = f"{role}-off-reference.png"
            before = checked_rgb(root / name, row["artifacts"][name])
            after = checked_rgb(BASE / SOURCE / f"{key}-{role}-reference.png",
                                source["qualification"][key]["reference_diagnostics"][role]["edited_sha256"])
            if before.shape != after.shape or before.shape[:2] != ref_person.shape:
                raise ValueError("Changed reference geometry")
            table, _ = paired_fit(before, after)
            global_render = apply(target, table)
            entry = source["comparison"]["paired"]["cases"][key]
            baseline = checked_rgb(BASE / SOURCE / f"paired-{key}-{role}-on.png", entry["outputs"][f"{role}-on"])
            np.testing.assert_array_equal(global_render, baseline)
            parts, pixels = mappings(before, after, ref_person)
            fallback = [value is None for value in parts]
            if all(fallback):
                regional = baseline
            else:
                fg, bg = (apply_mapping(target, value if value is not None else table, 1) for value in parts)
                regional = finish_spatial(target, np.rint(opacity*fg + (1 - opacity)*bg).astype(np.uint8), .8)
            rows[role] = {"original": target, "global": baseline, "regional": regional, "expected": expected}
            receipts[role] = {"fitted_pixels": pixels, "fallback": fallback,
                              "sha256": save(out / f"{key}-{role}.png", regional)}
            for arm, image in (("global", baseline), ("regional", regional)):
                rendered[arm][role] = {"off": target, "on": image}
        scores = {}
        for arm, values in rendered.items():
            style = evaluate(target, {"off": target, "on": expected}, values, person)
            err = {role: {k: float(np.abs(item["on"].astype(np.int16) - expected)[mask].mean())
                          for k, mask in areas.items()} for role, item in values.items()}
            scores[arm] = {"style": style, "mae": err,
                           "passed": style["passed"] and all(max(z.values()) <= 3 for z in err.values())}
        cases[key] = {"scores": scores, "outputs": receipts,
                      "comparison_sha256": panel(out / f"{key}-comparison.png", rows)}
    counts = {arm: sum(row["scores"][arm]["passed"] for row in cases.values()) for arm in rendered}
    worst = max(row["scores"]["regional"]["mae"][k][zone] - row["scores"]["global"]["mae"][k][zone]
                for row in cases.values() for k in ("C1", "F1") for zone in ("person", "neutral"))
    json_to(out / "results.json", {"counts": counts, "worst_protected_regression": worst, "cases": cases,
                                  "passed": counts["regional"] >= 3 and worst <= 2,
                                  "source_sha256": SOURCE_SHA, "mask_evidence_sha256": HELD_SHA})
    print(counts, worst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
