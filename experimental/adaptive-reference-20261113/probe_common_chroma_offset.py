"""Test whether removing one shared color shift preserves deliberate grading."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_hdm_hdr_equal_luminance import (
    EXPOSURES, PREPARED_SHA, RENDER_SHA, SCENE, Y_WEIGHTS, digest, encode, json_to,
    linear_rgb, preclip_grade, verified_json, verified_rgb,
)
from probe_hdm_poker_material_roi import REGIONS, evaluate
from spatial_render import finish_spatial


FILM_SHA = "34fb0882f660aa8aac7cb7d72fd10e47572d49f78f0a1107f3d6e9f87f6f3b60"


def correct(original, prediction):
    source, graded = map(linear_rgb, (original, prediction))
    source_y, y = source @ Y_WEIGHTS, graded @ Y_WEIGHTS
    support = ((source_y >= .02) & (y >= .02)
               & (source.max(-1) < .98) & (graded.max(-1) < .98))
    count = int(support.sum())
    if count < 256:
        return prediction, {"support": count, "fallback": True, "quantized_y_error_max": 0.}
    offsets = graded[support] / y[support, None] - source[support] / source_y[support, None]
    offset = np.median(offsets, axis=0)
    offset -= offset @ Y_WEIGHTS
    chroma = graded - y[..., None] * (1 + offset)
    headroom = np.where(chroma > 0, 1 - y[..., None], y[..., None])
    scale = np.divide(headroom, np.abs(chroma), out=np.full_like(chroma, np.inf), where=chroma != 0)
    factor = np.minimum(scale.min(axis=-1), 1)
    clipped = y[..., None] + factor[..., None] * chroma
    np.testing.assert_allclose(clipped @ Y_WEIGHTS, y, atol=1e-12, rtol=0)
    corrected = encode(clipped)
    return corrected, {
        "support": count, "fallback": False, "offset": offset.tolist(),
        "gamut_limited_fraction": float((factor < 1).mean()),
        "quantized_y_error_max": float(np.abs(linear_rgb(corrected) @ Y_WEIGHTS - y).max()),
    }


def verified_image(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected Film A image: {path}")
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB8 image: {path}")
        return np.asarray(image)


def save_image(output, name, pixels, summary):
    path = output / f"{name}.png"
    Image.fromarray(pixels).save(path)
    summary["sha256"] = digest(path)
    if summary["quantized_y_error_max"] > .005:
        raise ValueError("Encoded output changed luminance beyond tolerance")


def poker(base, output):
    prepared = base / "hdm-hdr-2014-display-prepared"
    rendered = base / "hdm-hdr-2014-render"
    images = verified_json(prepared / "prepared.json", PREPARED_SHA)["cases"][SCENE]["images"]
    renders = verified_json(rendered / "results.json", RENDER_SHA)["scores"][SCENE]["exposures"]
    grade, _ = preclip_grade(base / "hdm-hdr-2014-paired-stills")
    author = verified_rgb(prepared / images["graded"]["file"], images["graded"]["sha256"])
    masks = {}
    for name, points in REGIONS.items():
        raster = Image.new("L", (512, 288))
        ImageDraw.Draw(raster).polygon(points, fill=255)
        masks[name] = np.asarray(raster) > 0
    rows = {}
    for exposure in EXPOSURES:
        original = verified_rgb(prepared / images[exposure]["file"], images[exposure]["sha256"])
        original_only, _ = correct(original, original)
        np.testing.assert_array_equal(original_only, original)
        prediction = verified_rgb(rendered / f"{SCENE}-{exposure}-13.png", renders[exposure]["output_sha256"])
        candidate, stats = correct(original, prediction)
        save_image(output, f"poker-{exposure}", candidate, stats)
        raw, prior, final = map(linear_rgb, (original, prediction, candidate))
        areas = {}
        for name, mask in masks.items():
            distances = evaluate(raw, prior, final, grade, mask)["distance"]
            distances["corrected"] = distances.pop("retained")
            areas[name] = distances
        passed = all(
            data["corrected"] <= (data["model"] * .8 if name.startswith("cloth") else data["model"] + .001)
            for name, data in areas.items()
        )
        rows[exposure] = {"correction": stats, "areas": areas, "passed": passed}
        if exposure == "unit":
            panel = (original, prediction, candidate, author)
    return rows, panel


def film(base, output):
    source = base / "arri-shared-look-cross-source-v1"
    manifest = verified_json(source / "results.json", FILM_SHA)
    rows, panels = {}, {}
    for name, scene in manifest["prepared"].items():
        original = verified_image(source / f"{name}-off.png", scene["images"]["off"])
        authored = verified_image(source / f"{name}-on.png", scene["images"]["on"])
        expected = finish_spatial(original, authored, .8)
        variants = {"known-look": expected}
        for role, sha in manifest["predictions"][name]["hashes"].items():
            variants[role] = verified_image(source / f"{name}-13-reference-{role}.png", sha)
        corrected, settings = {}, {}
        for role, image in variants.items():
            corrected[role], settings[role] = correct(original, image)
            save_image(output, f"{name}-{role}", corrected[role], settings[role])
        whole = np.ones(original.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (original, authored)))
        assert int(neutral.sum()) == scene["shared_neutral_pixels"]
        scores, controls = {}, {}
        for key, region in (("whole", whole), ("shared_neutral", neutral)):
            scores[key] = metrics(original, expected, corrected["off"], corrected["on"], region)
            controls[key] = metrics(original, expected, original, corrected["known-look"], region)
        rows[name] = {
            "corrections": settings, "response": scores, "known_look_control": controls,
            "passed": all(row["passed"] for row in scores.values())
            and all(row["passed"] and row["on_matches_authored_mae"] <= 1 for row in controls.values()),
        }
        panels[name] = (original, variants["on"], corrected["on"], expected)
    return rows, panels


def run(base, output):
    output.mkdir()
    poker_rows, author_panel = poker(base, output)
    film_rows, film_panels = film(base, output)
    panel = Image.new("RGB", (1040, 480), "#222730")
    draw = ImageDraw.Draw(panel)
    for row, (scene, images) in enumerate({"Poker author": author_panel, **film_panels}.items()):
        for col, (arm, rgb) in enumerate(zip(("input", "13", "corrected", "expected"), images, strict=True)):
            x, y = col * 260 + 4, row * 160
            draw.text((x, y + 4), f"{scene} / {arm}", fill="white")
            tile = Image.fromarray(rgb)
            tile.thumbnail((250, 140), resample=Image.Resampling.LANCZOS)
            panel.paste(tile, (x, y + 20))
    panel.save(output / "comparison.png")
    json_to(output / "results.json", {
        "poker": poker_rows, "film_a": film_rows, "film_input_sha256": FILM_SHA,
        "poker_input_sha256": PREPARED_SHA, "poker_render_sha256": RENDER_SHA,
        "passed": all(row["passed"] for row in (*poker_rows.values(), *film_rows.values())),
        "comparison_sha256": digest(output / "comparison.png"),
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.base, args.output)
