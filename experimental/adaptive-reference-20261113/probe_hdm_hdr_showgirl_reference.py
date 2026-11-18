"""Render the fixed Showgirl 2 reference switch and score material responses offline."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_hdm_hdr_display import REC2020_TO_REC709, SOURCE_TO_REC709, decode_pq, digest, display, json_to
from probe_hdm_hdr_equal_luminance import linear_rgb, verified_rgb
from probe_hdm_hdr_showgirl_light import CASES, REGIONS, SIZE, material_measurement

MODEL_SHA = "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9"
ROLES = ("original", "graded")
STRENGTH = .8


def measure_display(pixels, box):
    x0, y0, x1, y1 = box
    region = pixels[y0:y1, x0:x1]
    result = material_measurement(linear_rgb(pixels), box)
    result["endpoint_fraction"] = float(np.any((region == 0) | (region == 255), axis=-1).mean())
    return result


def prepare(source, rois, output):
    import cv2
    from audit_hdm_hdr_still_alignment import verified_image

    cv2.setNumThreads(1)
    appendix = rois.read_bytes().replace(b"\r\n", b"\n")
    markings = json.loads(appendix)
    path = source / "manifest.json"
    if digest(path) != markings["manifest_sha256"]:
        raise ValueError("Unexpected archive manifest")
    manifest = json.loads(path.read_bytes())
    output.mkdir()
    cases = {}
    for case in CASES:
        boxes = {key: markings["cases"][case][key] for key in REGIONS}
        cases[case] = {"boxes": boxes}
        files = manifest["cases"][case]["files"]
        for role in ROLES:
            bgr = verified_image(source, files[role])
            if role == "original":
                frame, matrix = bgr[..., ::-1], SOURCE_TO_REC709
            else:
                frame, matrix = decode_pq(bgr[..., ::-1]), REC2020_TO_REC709
            linear = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA) @ matrix
            pixels, clipping = display(linear)
            item = output / f"{case}-{role}.png"
            Image.fromarray(pixels).save(item)
            cases[case][role] = {
                "sha256": digest(item), "source_sha256": files[role]["sha256"], "clipping": clipping,
                "display": {key: measure_display(pixels, box) for key, box in boxes.items()},
                "preclip": {key: material_measurement(linear, box) for key, box in boxes.items()},
            }
    json_to(output / "prepared.json", {"archive_sha256": digest(path), "roi_sha256": sha256(appendix).hexdigest(),
                                       "cases": cases})


def delta(after, before):
    return np.subtract(after["median_uv"], before["median_uv"])


def alignment(change, expected):
    length = float(np.linalg.norm(change))
    standard = float(np.linalg.norm(expected))
    dot = float(change @ expected)
    return {
        "change_uv": change.tolist(), "expected_uv": expected.tolist(), "change_length": length,
        "expected_length": standard, "cosine": dot / (length * standard) if length and standard else None,
        "projection": dot / standard**2 if standard else None,
        "length_ratio": length / standard if standard else None,
    }


def reference_response(frames, before, after, region):
    original, grade = (frames[key]["display"][region] for key in ROLES)
    expected = delta(grade, original)
    preclip = delta(frames["graded"]["preclip"][region], frames["original"]["preclip"][region])
    edit_alignment = alignment(delta(after, before), expected)
    display_agreement = alignment(expected, preclip)["cosine"]
    endpoints = grade["endpoint_fraction"]
    distance_before = float(np.linalg.norm(delta(before, grade)))
    distance_after = float(np.linalg.norm(delta(after, grade)))
    passes_in_display = (
        edit_alignment["expected_length"] >= .005
        and edit_alignment["cosine"] is not None and edit_alignment["cosine"] >= .5
        and .25 <= edit_alignment["projection"] <= 1.75
        and distance_after < distance_before
    )
    supported = endpoints <= .1 and display_agreement is not None and display_agreement >= .7
    return {
        **edit_alignment, "distance_to_grade_before": distance_before, "distance_to_grade_after": distance_after,
        "grade_endpoint_fraction": endpoints, "edit_display_preclip_cosine": display_agreement,
        "after_display_endpoint_fraction": after["endpoint_fraction"], "supported_display": supported,
        "passed_in_display": passes_in_display, "passed": passes_in_display and supported,
    }


def render(prepared, output):
    import torch
    from spatial_render import SpatialEngine, finish_spatial

    torch.set_num_threads(4)
    path = prepared / "prepared.json"
    records = json.loads(path.read_bytes())["cases"]
    photos = {}
    for case, role in product(CASES, ROLES):
        photos[case, role] = verified_rgb(prepared / f"{case}-{role}.png", records[case][role]["sha256"])
    model = SpatialEngine("multiref-v1")
    if model.checkpoint_sha != MODEL_SHA:
        raise ValueError("Unexpected 13 checkpoint")

    output.mkdir()
    board = Image.new("RGB", (6 * 268, 4 * 174), "#222730")
    draw = ImageDraw.Draw(board)
    outputs, comparisons = {}, {}
    for index, (target, guide) in enumerate(product(CASES, CASES)):
        raw = photos[target, "original"]
        captures = {}
        for role in ROLES:
            edited = finish_spatial(raw, model.prepare(raw, photos[guide, role]).apply(raw), STRENGTH)
            name = f"{target[-3:]}-ref-{guide[-3:]}-{role}"
            dest = output / f"{name}.png"
            Image.fromarray(edited).save(dest)
            captures[role] = edited
            outputs[target, guide, role] = {
                "file": dest.name, "sha256": digest(dest),
                "regions": {key: measure_display(edited, records[target]["boxes"][key]) for key in REGIONS},
                "mean_rgb_change_from_input": float(np.mean(np.abs(raw.astype(np.int16) - edited))),
            }
        arms = (
            ("target input", raw), ("reference source", photos[guide, "original"]),
            ("reference grade", photos[guide, "graded"]), ("13 / source reference", captures["original"]),
            ("13 / graded reference", captures["graded"]), ("archival target grade", photos[target, "graded"]),
        )
        for col, (label, pixels) in enumerate(arms):
            left, top = 268 * col + 6, 174 * index + 24
            draw.text((left, top - 19), f"{target[-3:]} / {guide[-3:]}  {label}", fill="white")
            board.paste(Image.fromarray(pixels).resize((256, 144), Image.Resampling.LANCZOS), (left, top))
        before, after = (outputs[target, guide, key]["regions"] for key in ROLES)
        comparisons[f"{target[-3:]}-ref-{guide[-3:]}"] = {
            "cross_time": target != guide,
            "regions": {key: reference_response(records[target], before[key], after[key], key) for key in REGIONS},
        }

    lighting = {}
    first, second = CASES
    for guide, role in product(CASES, ROLES):
        first_out, second_out = (outputs[target, guide, role]["regions"] for target in CASES)
        row = {}
        for region in REGIONS:
            input_delta = delta(records[second]["original"]["display"][region],
                                records[first]["original"]["display"][region])
            metrics = alignment(delta(second_out[region], first_out[region]), input_delta)
            metrics["passed"] = (
                metrics["cosine"] is not None and metrics["cosine"] >= .8 and .6 <= metrics["length_ratio"] <= 1.4
            )
            row[region] = metrics
        lighting[f"{guide[-3:]}-{role}"] = row

    board.save(output / "comparison.png")
    json_to(output / "results.json", {
        "prepared_sha256": digest(path), "model_sha256": model.checkpoint_sha, "strength": STRENGTH,
        "comparison_sha256": digest(output / "comparison.png"),
        "outputs": {row["file"]: row for row in outputs.values()},
        "reference_response": comparisons, "lighting_difference": lighting,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="stage", required=True)
    pre = stages.add_parser("prepare")
    pre.add_argument("source", type=Path)
    pre.add_argument("regions", type=Path)
    pre.add_argument("output", type=Path)
    run = stages.add_parser("render")
    run.add_argument("prepared", type=Path)
    run.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.stage == "prepare":
        prepare(args.source, args.regions, args.output)
    else:
        render(args.prepared, args.output)
