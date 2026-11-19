"""Attribute archived regional Film B failures to reference color support."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree

from probe_film_b_paired_regions import (
    BASE, HELD_SHA, PARENT, PARENT_SHA, ROOT, SOURCE, SOURCE_SHA, checked_rgb,
    digest, evaluate, inside, json_to, manifest, mask_for, panel, save, target_regions,
    thumbnail_pair, verified_json,
)

PREVIOUS = "film-b-paired-regions-v1"
EVIDENCE = "docs/research/2026-11-19-film-b-paired-regions-evidence.json"
RADIUS = 6.


def support(target, before, after, target_mask, ref_mask):
    small, edited = thumbnail_pair(before, after)
    h, w = small.shape[:2]
    rm = np.asarray(Image.fromarray(ref_mask).resize((w, h), Image.Resampling.NEAREST))
    yy, xx = np.indices((h, w))
    valid = ((xx//8 + yy//8) % 2 == 0) & ((edited > 2) & (edited < 253)).all(axis=-1)
    limits = []
    for region in (rm, ~rm):
        keys = np.unique(small[inside(region) & valid], axis=0)
        if not len(keys):
            raise ValueError("Expected prior fit support")
        d, _ = cKDTree(keys).query(target.reshape(-1, 3), workers=1)
        limits.append(d.reshape(target.shape[:2]) <= RADIUS)
    opacity = np.asarray(Image.fromarray(target_mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(2)))
    return ((opacity == 0) | limits[0]) & ((opacity == 255) | limits[1])


def region_error(area, *errors):
    return {"pixels": int(area.sum()),
            "mae": [float(error[area].mean()) for error in errors] if area.any() else None}


def run(out):
    receipt = json.loads((ROOT / EVIDENCE).read_bytes())
    old = verified_json(BASE / PREVIOUS / "results.json", receipt.pop("report_sha256"))
    if old != receipt:
        raise ValueError("Previous comparison changed")
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    held = verified_json(ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json", HELD_SHA)
    refs = {row["target_pair"]: row["input"] for row in held["cases"].values()}
    parents = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    out.mkdir()
    cases = {}
    for key, data in manifest(BASE, "bg-real-editor-look")["cases"].items():
        root = BASE / "bg-real-editor-look" / key
        qualification = prior["qualification"][key]
        target = checked_rgb(root / "target.png", data["artifacts"]["target.png"])
        expected = checked_rgb(BASE / SOURCE / f"{key}-expected.png", qualification["expected_sha256"])
        person = mask_for(parents[key], key.split("-")[0])
        masks = target_regions(target, person)
        areas = {"person": person, "neutral": masks["neutral_proxy"], "whole": masks["whole"]}
        mask_entry = refs[key]["mask"]
        path = BASE / "bg-real-pixel-inputs" / mask_entry["name"]
        if digest(path) != mask_entry["sha256"]:
            raise ValueError("Reference mask changed")
        with Image.open(path) as im:
            rm = np.asarray(im.convert("L")) >= 128
        record, board, pictures = {}, {}, {}
        for role in ("C1", "F1"):
            stem = f"{key}-{role}"
            before = checked_rgb(root / f"{role}-off-reference.png", data["artifacts"][f"{role}-off-reference.png"])
            after = checked_rgb(BASE / SOURCE / f"{stem}-reference.png",
                                qualification["reference_diagnostics"][role]["edited_sha256"])
            control = checked_rgb(BASE / SOURCE / f"paired-{stem}-on.png",
                                  prior["comparison"]["paired"]["cases"][key]["outputs"][f"{role}-on"])
            row = old["cases"][key]
            if any(row["outputs"][role]["fallback"]):
                raise ValueError("Unexpected original fitting fallback")
            candidate = checked_rgb(BASE / PREVIOUS / f"{stem}.png", row["outputs"][role]["sha256"])
            use = support(target, before, after, person, rm)
            mixed = np.where(use[..., None], candidate, control)
            rgb = (control, candidate, mixed)
            errors = [np.abs(image.astype(np.int16) - expected) for image in rgb]
            groups = {}
            for zone, mask in areas.items():
                groups[zone] = {"supported": region_error(mask & use, *errors),
                                "unsupported": region_error(mask & ~use, *errors),
                                "mae": [float(e[mask].mean()) for e in errors]}
                assert groups[zone]["mae"][:2] == [row["scores"][arm]["mae"][role][zone]
                                                   for arm in ("global", "regional")]
            record[role] = {"groups": groups, "sha256": save(out / f"{stem}.png", mixed)}
            board[role] = {"original": target, "global": control, "regional": candidate, "coverage fallback": mixed,
                           "expected": expected}
            pictures[role] = {"off": target, "on": mixed}
        result = evaluate(target, {"off": target, "on": expected}, pictures, person)
        absolute = [x["mae"][2] for v in record.values() for x in v["groups"].values()]
        cases[key] = {"outputs": record, "style": result, "passed": bool(result["passed"] and max(absolute) <= 3),
                      "comparison_sha256": panel(out / f"{key}-comparison.png", board)}
    worst = max(g["mae"][2]-g["mae"][0] for case in cases.values() for row in case["outputs"].values()
                for k,g in row["groups"].items() if k in ("person", "neutral"))
    count = sum(case["passed"] for case in cases.values())
    data = [v["groups"]["person"]["supported"] for case in cases.values() for v in case["outputs"].values()]
    visible = sum(d["pixels"] for d in data)
    if not visible:
        raise ValueError("No supported person pixels")
    mean = [sum(d["pixels"]*d["mae"][i] for d in data if d["pixels"]) / visible for i in range(2)]
    report = {"cases": cases, "passing_cases": count, "worst_regression": worst,
              "supported_person_mae": mean, "passed": count >= 3 and worst <= 2 and mean[1] <= mean[0],
              "previous_sha256": digest(BASE / PREVIOUS / "results.json"), "radius": RADIUS}
    json_to(out / "results.json", report)
    print(count, worst, mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
