"""Attribute an observed neutral cast to the reference source and its grade."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from probe_hdm_bistro import raw_image, save_rgb
from probe_hdm_hdr_display import EXPOSURES, REC2020_TO_REC709, decode_pq
from probe_hdm_hdr_equal_luminance import (
    PREPARED_SHA, RENDER_SHA, Y_WEIGHTS, encode, json_to, linear_rgb, match_y,
    preclip_grade, uv, verified_json, verified_rgb,
)
from spatial_render import SpatialEngine, finish_spatial


BISTRO_ARCHIVE = "dd7e89857a5215bab51c71b53880e971fa5b5a954bab050ceeb0ea51db4bbe3c"


def original_set(folder, records):
    return {key: verified_rgb(folder / record["file"], record["sha256"])
            for key, record in records.items()}


def targets(root, prepared, prep):
    poker_record = prep["cases"]["poker_fullshot"]["images"]
    poker_inputs = original_set(prepared, poker_record)
    poker_truth, _ = preclip_grade(root / "hdm-hdr-2014-paired-stills")
    poker_output = root / "hdm-hdr-2014-render"
    poker_sha = verified_json(poker_output / "results.json", RENDER_SHA)["scores"]["poker_fullshot"]["exposures"]["unit"]["output_sha256"]
    yield ("poker", poker_inputs, poker_truth,
           verified_rgb(poker_output / "poker_fullshot-unit-13.png", poker_sha))

    archive = root / "hdm-bistro-frames-v1"
    cases = verified_json(archive / "manifest.json", BISTRO_ARCHIVE)["cases"]
    raw_grade = raw_image(archive, cases, "bistro", "graded")
    bistro_truth = cv2.resize(decode_pq(raw_grade[..., ::-1]), (512, 288), interpolation=cv2.INTER_AREA) @ REC2020_TO_REC709
    bistro_output = root / "hdm-bistro-pilot-v1"
    bistro = json.loads((bistro_output / "results.json").read_bytes())
    if bistro["source_sha256"] != BISTRO_ARCHIVE or not bistro["alignment"]["qualified"]:
        raise ValueError("Unexpected Bistro source alignment")
    bistro_render = bistro["render"]
    bistro_inputs = {
        key: verified_rgb(bistro_output / f"{key}-input.png", row["input_sha256"])
        for key, row in bistro_render["exposures"].items()
    }
    bistro_inputs["graded"] = verified_rgb(bistro_output / "author-sdr.png", bistro_render["author_sha256"])
    yield ("bistro", bistro_inputs, bistro_truth,
           verified_rgb(bistro_output / "unit-13.png", bistro_render["exposures"]["unit"]["model_sha256"]))


def measure(images, truth, neutral):
    if neutral.sum() < 256:
        raise ValueError("Insufficient common neutral coverage")
    points = {key: uv(linear_rgb(pixels)[neutral]) for key, pixels in images.items()}
    expected = uv(truth[neutral])
    distances = {key: float(np.linalg.norm(value - expected, axis=1).mean())
                 for key, value in points.items()}
    total = points["grade_ref"] - points["target"]
    segments = {
        "raw_reference": points["raw_ref"] - points["target"],
        "matched_light_increment": points["matched_ref"] - points["raw_ref"],
        "matched_color_increment": points["grade_ref"] - points["matched_ref"],
    }
    norm = float(np.sum(total**2))
    if norm <= 0:
        raise ValueError("No observable displacement")
    projections = {key: float(np.sum(delta * total) / norm) for key, delta in segments.items()}
    np.testing.assert_allclose(sum(projections.values()), 1., atol=1e-12, rtol=0)
    return {
        "neutral_pixels": int(neutral.sum()), "distance_to_author_preclip_uv": distances,
        "total_change_projection": projections,
        "raw_reference_already_propagates_cast": projections["raw_reference"] > .5
        and distances["raw_ref"] - distances["target"] >= .001,
    }


def run(root, out):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    prepared = root / "hdm-hdr-2014-display-prepared"
    prep = verified_json(prepared / "prepared.json", PREPARED_SHA)
    guide_input = original_set(prepared, prep["cases"]["showgirl_01"]["images"])
    raw, grade = guide_input["unit"], guide_input["graded"]
    y = linear_rgb(grade) @ Y_WEIGHTS
    matched_linear, limited = match_y(linear_rgb(raw), y)
    np.testing.assert_allclose(matched_linear @ Y_WEIGHTS, y, atol=1e-12, rtol=0)
    matched = encode(matched_linear)
    y_error = float(np.abs(linear_rgb(matched) @ Y_WEIGHTS - y).max())
    limited_fraction = float(limited.mean())
    support = y_error <= .005 and limited_fraction <= .10
    guides = {"raw_ref": raw, "matched_ref": matched, "grade_ref": grade}
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9":
        raise ValueError("Unexpected 13 checkpoint")
    out.mkdir()
    rows = {}
    board = Image.new("RGB", (1350, 3 * 176), "#222730")
    draw = ImageDraw.Draw(board)
    draw_tiles(board, draw, 0, "reference", guides)
    for index, (name, inputs, truth, prior) in enumerate(targets(root, prepared, prep), 1):
        target = inputs["unit"]
        composed = {key: finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                    for key, guide in guides.items()}
        np.testing.assert_array_equal(composed["grade_ref"], prior)
        neutral = np.logical_and.reduce([
            (np.ptp(inputs[key], axis=-1) <= 12) & (inputs[key].min(-1) >= 32) & (inputs[key].max(-1) <= 240)
            for key in EXPOSURES
        ])
        rows[name] = measure({"target": target, **composed}, truth, neutral)
        rows[name]["outputs_sha256"] = {key: save_rgb(out, f"{name}-{key}", image)
                                        for key, image in composed.items()}
        row = {"target": target, **composed, "author": inputs["graded"]}
        draw_tiles(board, draw, index, name, row)
    json_to(out / "results.json", {
        "weight_sha256": engine.checkpoint_sha, "prepared_sha256": PREPARED_SHA,
        "reference_raw_sha256": prep["cases"]["showgirl_01"]["images"]["unit"]["sha256"],
        "reference_grade_sha256": prep["cases"]["showgirl_01"]["images"]["graded"]["sha256"],
        "matched_sha256": save_rgb(out, "reference-matched-light", matched),
        "matched_y_max_error": y_error, "matched_gamut_limited_fraction": limited_fraction,
        "matched_arm_qualified": support, "targets": rows,
    })
    board.save(out / "comparison.png")


def draw_tiles(board, draw, row, prefix, images):
    columns = ("target", "raw_ref", "matched_ref", "grade_ref", "author")
    for key, pixels in images.items():
        x, y = columns.index(key) * 270 + 7, row * 176
        draw.text((x, y + 6), f"{prefix} / {key}", fill="white")
        board.paste(Image.fromarray(pixels).resize((256, 144), Image.Resampling.LANCZOS), (x, y + 26))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.output)
