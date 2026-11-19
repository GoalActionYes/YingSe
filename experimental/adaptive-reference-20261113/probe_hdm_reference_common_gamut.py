"""Cross reference luminance and chromaticity in their shared SDR gamut."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from probe_hdm_bistro import save_rgb
from probe_hdm_hdr_display import EXPOSURES
from probe_hdm_hdr_equal_luminance import PREPARED_SHA, Y_WEIGHTS, digest, encode, json_to, linear_rgb, uv, verified_json
from probe_hdm_reference_edit_toggle import original_set, targets
from spatial_render import SpatialEngine, finish_spatial


ORDER = ("low_raw", "high_raw", "low_grade", "high_grade")


def prepare_references(raw, grade):
    first, last = map(linear_rgb, (raw, grade))
    raw_y, grade_y = (image @ Y_WEIGHTS for image in (first, last))
    ratios = [np.divide(image, y[..., None], out=np.zeros_like(image), where=y[..., None] > 0)
              for image, y in ((first, raw_y), (last, grade_y))]
    limits = np.maximum(ratios[0].max(-1), ratios[1].max(-1))
    cap = np.divide(1, limits, out=np.zeros_like(limits), where=limits > 0)
    low = np.minimum(raw_y, grade_y)
    high = np.minimum(np.maximum(raw_y, grade_y), cap)
    high = np.where(low > 0, high, 0)
    np.testing.assert_array_less(low - 1e-12, high)
    floating = dict(zip(ORDER, (low[..., None] * ratios[0], high[..., None] * ratios[0],
                                low[..., None] * ratios[1], high[..., None] * ratios[1]), strict=True))
    for image in floating.values():
        if image.min() < 0 or image.max() > 1 + 1e-12:
            raise ValueError("Common-gamut construction failed")
    guides = {name: encode(image) for name, image in floating.items()}
    linear = {name: linear_rgb(image) for name, image in guides.items()}
    y_error = max(float(np.abs((linear[f"{level}_raw"] - linear[f"{level}_grade"]) @ Y_WEIGHTS).max())
                  for level in ("low", "high"))
    support = low >= .01
    coverage = float(support.mean())
    stats = {"support_fraction": coverage, "matched_y_max_difference": y_error}
    if not support.any():
        return guides, stats | {"qualified": False}
    points = {name: uv(image[support]) for name, image in linear.items()}
    y_gap = min(float(((linear[f"high_{color}"] - linear[f"low_{color}"]) @ Y_WEIGHTS)[support].mean())
                for color in ("raw", "grade"))
    color_gap = min(float(np.median(np.linalg.norm(points[f"{level}_grade"] - points[f"{level}_raw"], axis=1)))
                    for level in ("low", "high"))
    color_drift = max(float(np.median(np.linalg.norm(points[f"high_{color}"] - points[f"low_{color}"], axis=1)))
                      for color in ("raw", "grade"))
    stats.update(mean_y_gap_min=y_gap, color_gap_median_min=color_gap, color_drift_median_max=color_drift,
                 qualified=coverage >= .25 and y_error <= .005 and y_gap >= .02
                 and color_gap >= .005 and color_drift <= .002)
    return guides, stats


def measure(inputs, outputs, truth):
    neutral = np.logical_and.reduce([
        (np.ptp(inputs[key], axis=-1) <= 12) & (inputs[key].min(-1) >= 32) & (inputs[key].max(-1) <= 240)
        for key in EXPOSURES
    ])
    if neutral.sum() < 256:
        raise ValueError("Insufficient common neutral coverage")
    points = {name: uv(linear_rgb(image)[neutral]) for name, image in (outputs | {"target": inputs["unit"]}).items()}
    expected = uv(truth[neutral])
    distances = {name: float(np.linalg.norm(value - expected, axis=1).mean()) for name, value in points.items()}
    bright_raw = points["high_raw"] - points["low_raw"]
    bright_grade = points["high_grade"] - points["low_grade"]
    color_low = points["low_grade"] - points["low_raw"]
    color_high = points["high_grade"] - points["high_raw"]
    direction = points["high_grade"] - points["low_raw"]
    denominator = float(np.sum(direction**2))
    if denominator <= 0:
        raise ValueError("No observable diagonal displacement")
    contrasts = {"brightness_at_raw_color": bright_raw, "brightness_at_grade_color": bright_grade,
                 "color_at_low_brightness": color_low, "color_at_high_brightness": color_high,
                 "brightness_main": (bright_raw + bright_grade) / 2, "color_main": (color_low + color_high) / 2}
    projected = {name: float(np.sum(effect * direction) / denominator) for name, effect in contrasts.items()}
    np.testing.assert_allclose(projected["brightness_main"] + projected["color_main"], 1., atol=1e-12, rtol=0)
    return {"neutral_pixels": int(neutral.sum()), "author_distances": distances, "projections": projected,
            "interaction_mean_uv_norm": float(np.linalg.norm(bright_grade - bright_raw, axis=1).mean()),
            "brightness_dominant": projected["brightness_main"] >= 2/3 and
            min(projected["brightness_at_raw_color"], projected["brightness_at_grade_color"]) > .5}


def draw_row(board, draw, index, label, images):
    for col, (name, image) in enumerate(images.items()):
        x, y = col * 270 + 7, index * 174
        draw.text((x, y + 5), f"{label} / {name}", fill="white")
        board.paste(Image.fromarray(image).resize((256, 144), Image.Resampling.LANCZOS), (x, y + 23))


def run(base, out):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    prepared = base / "hdm-hdr-2014-display-prepared"
    prep = verified_json(prepared / "prepared.json", PREPARED_SHA)
    photos = original_set(prepared, prep["cases"]["showgirl_01"]["images"])
    guides, support = prepare_references(photos["unit"], photos["graded"])
    out.mkdir()
    board = Image.new("RGB", (1080, 522 if support["qualified"] else 174), "#222730")
    draw = ImageDraw.Draw(board)
    draw_row(board, draw, 0, "reference", guides)
    rows = {}
    weight_sha256 = None
    if support["qualified"]:
        engine = SpatialEngine("multiref-v1")
        weight_sha256 = engine.checkpoint_sha
        if weight_sha256 != "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9":
            raise ValueError("Unexpected 13 checkpoint")
        for index, (name, inputs, truth, _) in enumerate(targets(base, prepared, prep), 1):
            target = inputs["unit"]
            outputs = {key: finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
                       for key, ref in guides.items()}
            rows[name] = measure(inputs, outputs, truth)
            rows[name]["output_sha256"] = {key: save_rgb(out, f"{name}-{key}", image) for key, image in outputs.items()}
            draw_row(board, draw, index, name, outputs)
    board.save(out / "comparison.png")
    json_to(out / "results.json", {
        "prepared_sha256": PREPARED_SHA, "weight_sha256": weight_sha256,
        "reference_support": support, "targets": rows,
        "both_brightness_dominant": len(rows) == 2 and all(row["brightness_dominant"] for row in rows.values()),
        "references_sha256": {name: save_rgb(out, f"reference-{name}", image) for name, image in guides.items()},
        "comparison_sha256": digest(out / "comparison.png"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.output)
