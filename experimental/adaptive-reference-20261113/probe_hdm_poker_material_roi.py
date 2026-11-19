"""Locate the Poker chromaticity failure on fixed, source-only material ROIs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from probe_hdm_hdr_equal_luminance import (
    EXPOSURES, PREPARED_SHA, RENDER_SHA, SCENE, Y_WEIGHTS, digest, json_to,
    linear_rgb, preclip_grade, uv, verified_json, verified_rgb,
)


REGIONS = {
    "cloth_left": ((203, 136), (211, 136), (210, 166), (201, 169)),
    "cloth_right": ((292, 137), (305, 137), (302, 169), (301, 188), (290, 181)),
    "shirt": ((274, 83), (289, 81), (295, 94), (291, 109), (275, 111)),
    "forehead": ((256, 44), (262, 39), (267, 40), (271, 45), (270, 53), (257, 51)),
}


def evaluate(raw, model, retained, expected, mask):
    common = mask & np.logical_and.reduce(
        [(image @ Y_WEIGHTS) > 0 for image in (raw, model, retained, expected)]
    )
    reference = uv(expected[common])
    distances = {
        name: np.linalg.norm(uv(image[common]) - reference, axis=1)
        for name, image in (("input", raw), ("model", model), ("retained", retained))
    }
    increase = distances["model"] - distances["input"]
    return {
        "pixels": int(common.sum()),
        "distance": {name: float(values.mean()) for name, values in distances.items()},
        "model_increase": float(increase.mean()),
        "worsened_fraction": float((increase > 0).mean()),
        "retained_vs_model_mean_abs_y": float(
            np.abs((retained[common] - model[common]) @ Y_WEIGHTS).mean()
        ),
    }


def make_panel(views, path):
    board = Image.new("RGB", (1088, 240), "#222730")
    labels = ("Input and ROIs", "Input color / 13 Y", "13 (80%)", "Archival SDR preview")
    for index, (label, pixels) in enumerate(zip(labels, views, strict=True)):
        image = Image.fromarray(pixels)
        if index == 0:
            overlay = ImageDraw.Draw(image)
            for name, points in REGIONS.items():
                overlay.line((*points, points[0]), fill="#61e2ed" if name.startswith("cloth") else "#f9bf65")
        crop = image.crop((135, 20, 390, 224))
        board.paste(crop, (index * 272 + 8, 27))
        ImageDraw.Draw(board).text((index * 272 + 8, 8), label, fill="white")
    board.save(path)


def run(base, output):
    prepared = base / "hdm-hdr-2014-display-prepared"
    render = base / "hdm-hdr-2014-render"
    control = base / "hdm-hdr-poker-chroma-v1"
    images = verified_json(prepared / "prepared.json", PREPARED_SHA)["cases"][SCENE]["images"]
    models = verified_json(render / "results.json", RENDER_SHA)["scores"][SCENE]["exposures"]
    prior = json.loads((control / "results.json").read_bytes())
    if (prior["prepared_sha256"], prior["render_sha256"]) != (PREPARED_SHA, RENDER_SHA):
        raise ValueError("Unexpected equal-luminance run")
    grade, grade_sha = preclip_grade(base / "hdm-hdr-2014-paired-stills")
    if grade_sha != prior["pq_grade_sha256"]:
        raise ValueError("Unexpected archival frame")
    masks = {}
    for name, points in REGIONS.items():
        raster = Image.new("L", (512, 288))
        ImageDraw.Draw(raster).polygon(points, fill=255)
        masks[name] = {"full": np.asarray(raster) > 0,
                       "eroded": np.asarray(raster.filter(ImageFilter.MinFilter(5))) > 0}

    output.mkdir()
    results = {}
    for exposure in EXPOSURES:
        original = verified_rgb(prepared / images[exposure]["file"], images[exposure]["sha256"])
        result = verified_rgb(render / f"{SCENE}-{exposure}-13.png", models[exposure]["output_sha256"])
        retained = verified_rgb(control / f"poker-{exposure}-input-color-model-y.png",
                                prior["rows"][exposure]["panel"]["sha256"])
        raw, model, matched = map(linear_rgb, (original, result, retained))
        results[exposure] = {
            name: {key: evaluate(raw, model, matched, grade, mask) for key, mask in variants.items()}
            for name, variants in masks.items()
        }
        if exposure == "unit":
            author = verified_rgb(prepared / images["graded"]["file"], images["graded"]["sha256"])
            make_panel((original, retained, result, author), output / "comparison.png")
    passed = all(
        summary["pixels"] >= 80 and summary["model_increase"] >= .001
        for exposure in results.values() for name in ("cloth_left", "cloth_right")
        for summary in exposure[name].values()
    )
    json_to(output / "results.json", {
        "prepared_sha256": PREPARED_SHA,
        "render_sha256": RENDER_SHA,
        "pq_grade_sha256": grade_sha,
        "regions": REGIONS,
        "exposures": results,
        "cloth_localization_passed": passed,
        "comparison_sha256": digest(output / "comparison.png"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.output)
