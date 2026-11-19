"""Measure the separate SDR reference light and color responses of 13."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_facd_endpoint_support import interior
from probe_facd_two_image_baseline import CHECKPOINT, STYLES, read_checked
from probe_hdm_hdr_equal_luminance import Y_WEIGHTS, digest, json_to, linear_rgb, uv
from probe_hdm_reference_common_gamut import ORDER, prepare_references
from spatial_render import SpatialEngine, finish_spatial


def score(target, expected, outputs):
    neutral = target_regions(target, np.zeros(target.shape[:2], dtype=bool))["neutral_proxy"]
    eligible = neutral & interior(target) & interior(expected)
    linear = {key: linear_rgb(image) for key, image in (outputs | {"input": target, "author": expected}).items()}
    mask = eligible & np.logical_and.reduce([(image @ Y_WEIGHTS) > 0 for image in linear.values()])
    row = {"eligible_neutral_pixels": int(eligible.sum()), "measured_pixels": int(mask.sum())}
    if mask.sum() < 256:
        return row | {"scorable": False, "reason": "insufficient pixels"}
    points = {key: uv(image[mask]) for key, image in linear.items()}
    desired = points["author"] - points["input"]
    norm = float(np.sum(desired**2))
    row["author_edit_rms"] = float(np.sqrt(norm / mask.sum()))
    row["distance_to_author"] = {key: float(np.linalg.norm(pts - points["author"], axis=1).mean())
                                 for key, pts in points.items() if key != "author"}
    if row["author_edit_rms"] < .005:
        return row | {"scorable": False, "reason": "weak authored chroma change"}
    bright_raw = points["high_raw"] - points["low_raw"]
    bright_grade = points["high_grade"] - points["low_grade"]
    light = (bright_raw + bright_grade) / 2
    color = ((points["low_grade"] - points["low_raw"]) + (points["high_grade"] - points["high_raw"])) / 2
    projection = {key: float(np.sum(delta * desired) / norm)
                  for key, delta in {"brightness": light, "color": color}.items()}
    return row | {"scorable": True, "projected_author_change": projection,
                  "interaction_mean_uv_norm": float(np.linalg.norm(bright_grade - bright_raw, axis=1).mean()),
                  "brightness_dominant": projection["brightness"] >= .5 and projection["brightness"] > projection["color"]}


def save_images(folder, prefix, images):
    hashes = {}
    for key, pixels in images.items():
        path = folder / f"{prefix}-{key}.png"
        Image.fromarray(pixels).save(path)
        hashes[key] = digest(path)
    return hashes


def panel(folder, guides, outputs, target, expected):
    board = Image.new("RGB", (1040, 432), "#242933")
    draw = ImageDraw.Draw(board)
    rows = (guides, outputs, {"target original": target, "author 80%": expected})
    for row, images in enumerate(rows):
        for col, (label, pixels) in enumerate(images.items()):
            x, y = col * 260 + 6, row * 144
            thumb = Image.fromarray(pixels)
            thumb.thumbnail((246, 120))
            draw.text((x, y + 4), f"{('reference', '13 (80%)', 'target')[row]} / {label}", fill="white")
            board.paste(thumb, (x, y + 20))
    path = folder / "comparison.png"
    board.save(path)
    return digest(path)


def run(aligned, evidence, assignment_path, out):
    source = json.loads(evidence.read_bytes())
    assignments = json.loads(assignment_path.read_bytes())
    groups = [key for key, case in source["cases"].items()
              if all(row["overlay"]["passed"] for row in case["checks"].values())]
    expected_assignments = {group: [groups[index-1], groups[(index+1) % len(groups)]]
                            for index, group in enumerate(groups)}
    if len(groups) != 7 or assignments != expected_assignments:
        raise ValueError("Unexpected fixed FACD reference assignments")
    torch.set_num_threads(4)
    out.mkdir()
    engine = None
    records = {}
    for group, (donor, _) in assignments.items():
        donor_original = read_checked(aligned, donor, "Origin", source)
        target = read_checked(aligned, group, "Origin", source)
        rows = {}
        for style in STYLES:
            donor_grade = read_checked(aligned, donor, style, source)
            guides, support = prepare_references(donor_original, donor_grade)
            summary = {"reference_support": support}
            rows[style] = summary
            if not support["qualified"]:
                continue
            if engine is None:
                engine = SpatialEngine("multiref-v1")
                if engine.checkpoint_sha != CHECKPOINT:
                    raise ValueError("Unexpected 13 model")
            folder = out / group / style
            folder.mkdir(parents=True)
            outputs = {key: finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                       for key, guide in guides.items()}
            summary["output_sha256"] = save_images(folder, "grade", outputs)
            summary["reference_sha256"] = save_images(folder, "reference", guides)
            expected = finish_spatial(target, read_checked(aligned, group, style, source), .8)
            summary["target_score"] = score(target, expected, outputs)
            summary["comparison_sha256"] = panel(folder, guides, outputs, target, expected)
        records[group] = {"donor": donor, "styles": rows}
    dominant_groups = {
        style: [group for group, case in records.items()
                if case["styles"][style].get("target_score", {}).get("brightness_dominant")]
        for style in STYLES
    }
    json_to(out / "results.json", {
        "alignment_sha256": digest(evidence), "assignment_sha256": digest(assignment_path),
        "weight_sha256": None if engine is None else engine.checkpoint_sha,
        "reference_arm_order": ORDER, "cases": records, "dominant_groups_by_style": dominant_groups,
        "brightness_dominant_across_content": any(len(scenes) >= 4 for scenes in dominant_groups.values()),
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned", "evidence", "assignment_path", "out"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
