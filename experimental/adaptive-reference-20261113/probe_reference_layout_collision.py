"""Check whether moving a fixed edit in the reference can change 13's render."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest
from probe_hdm_chart import CHECKPOINT
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_mls_film_light_cross import SOURCE_SHA
from spatial_render import SpatialEngine, finish_spatial


PARENT_SHA = "add480d14eeee078b9108c4a1575719199637010bdacc92f21de0717ffe34754"


def reference(base, known):
    def half(rgb):
        return np.asarray(Image.fromarray(rgb).resize((128, 256), Image.Resampling.BILINEAR))

    left = np.concatenate((half(known), half(base)[:, ::-1]), axis=1)
    return {"left": left, "right": left[:, ::-1].copy()}


def save(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def panel(path, refs, target, answers, results, output_label="13 output"):
    board = Image.new("RGB", (1000, 490), "#222730")
    draw = ImageDraw.Draw(board)
    for index, (side, guide) in enumerate(refs.items()):
        y = index * 245
        for col, (label, rgb) in enumerate(zip(
            (f"Reference / {side}", "Unchanged target", "Expected layout", output_label),
            (guide, target, answers[side], results[side]), strict=True,
        )):
            x = col * 250 + 5
            draw.text((x, y + 5), label, fill="white")
            tile = ImageOps.contain(Image.fromarray(rgb), (235, 220))
            board.paste(tile, (x, y + 20))
    board.save(path)
    return digest(path)


def distance(left, right):
    return float(np.abs(left.astype(np.int16) - right).mean())


@torch.inference_mode()
def run(base, out):
    source = base / "mls-linear-pilot"
    original = verified_json(source / "results.json", SOURCE_SHA)
    parent = base / "mls-film-light-cross-v1"
    crossed = verified_json(parent / "results.json", PARENT_SHA)
    if crossed["weight_sha256"] != CHECKPOINT:
        raise ValueError("Changed frozen model")
    images = {}
    for scene in ("02", "03"):
        filename = f"{scene}_2HAL_DESK.png"
        images[scene] = (
            verified(source / filename, original["scenes"][scene]["inputs_sha256"][filename]),
            verified(parent / f"{scene}-standard-reference-film.png", crossed["references"][scene]["standard"]),
        )
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed loaded model")
    out.mkdir()
    cases = {}
    for scene, donor in (("02", "03"), ("03", "02")):
        target, authored = images[scene]
        edited = finish_spatial(target, authored, .8)
        whole = np.ones(target.shape[:2], bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"]
                                   for rgb in (target, authored)))
        if neutral.sum() != crossed["cases"][scene]["neutral_pixels"]:
            raise ValueError("Changed neutral region")
        left = np.broadcast_to(np.arange(target.shape[1]) < target.shape[1] // 2, whole.shape)
        refs = reference(*images[donor])
        plans, outputs, answers, histograms, hashes = {}, {}, {}, {}, {}
        for side, guide in refs.items():
            region = left if side == "left" else ~left
            answers[side] = np.where(region[..., None], edited, target)
            small = engine.analysis(guide)
            histograms[side] = engine.api["batch_histogram_1D"](engine.api["srgb_to_ycbcr"](small))
            plans[side] = engine.prepare(target, guide)
            outputs[side] = finish_spatial(target, plans[side].apply(target), .8)
            hashes[side] = {
                label: save(out / f"{scene}-{side}-{label}.png", rgb)
                for label, rgb in (("reference", guide), ("expected", answers[side]), ("13", outputs[side]))
            }
        rgb_equal = np.array_equal(refs["left"][:, ::-1], refs["right"])
        histogram_difference = float((histograms["left"] - histograms["right"]).abs().max())
        grid_difference = float((plans["left"].grid - plans["right"].grid).abs().max())
        repeated = rgb_equal and histogram_difference == 0 and grid_difference == 0
        scores = {}
        for name, region in (("whole", whole), ("neutral", neutral)):
            separation = distance(answers["left"][region], answers["right"][region])
            output_difference = distance(outputs["left"][region], outputs["right"][region])
            scores[name] = {
                "pixels": int(region.sum()),
                "answer_difference_mae": separation,
                "shared_output_worst_mae_lower_bound": separation / 2,
                "output_difference_mae": output_difference,
                "errors": {side: distance(outputs[side][region], answers[side][region]) for side in refs},
                "incompatible_with_both_under_3": repeated and output_difference == 0 and separation > 6,
            }
        cases[scene] = {
            "reference_scene": donor, "exact_reference_mirror": rgb_equal,
            "histogram_max_difference": histogram_difference, "grid_max_difference": grid_difference,
            "outputs_identical": np.array_equal(outputs["left"], outputs["right"]),
            "scores": scores, "image_sha256": hashes,
            "panel_sha256": panel(out / f"{scene}-comparison.png", refs, target, answers, outputs),
        }
    json_to(out / "results.json", {"parent_sha256": PARENT_SHA, "weight_sha256": CHECKPOINT,
                                   "strength": .8, "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
