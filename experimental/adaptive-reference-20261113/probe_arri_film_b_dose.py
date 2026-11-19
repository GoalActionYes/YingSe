"""Measure 13's reference-dose response between frozen Film B endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_arri_shared_look import digest, metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_hdm_hdr_display import json_to
from spatial_render import SpatialEngine, finish_spatial


DOSES = (0, 25, 50, 75, 100)


def verified(path, sha):
    if digest(path) != sha:
        raise ValueError(f"Changed source: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def endpoints(root, report, scene):
    case = report["preparation"]["film_b"][scene]["images"]
    return tuple(verified(root / "film_b" / f"{scene}-{role}.png", case[role])
                 for role in ("off", "on"))


def response(target, expected, outputs, mask):
    initial = outputs[0][mask].astype(float)
    want = expected[100][mask].astype(float) - target[mask]
    power = np.vdot(want, want)
    if power == 0:
        return {"pixels": int(mask.sum()), "scored": False}
    rows = {}
    for amount, image in outputs.items():
        observed = image[mask].astype(float) - initial
        desired = expected[amount][mask].astype(float) - target[mask]
        rows[amount] = {"projection": float(np.vdot(observed, want) / power),
                        "response_mae": float(np.abs(observed - desired).mean())}
    projection = [row["projection"] for row in rows.values()]
    return {
        "pixels": int(mask.sum()), "scored": True, "rows": rows,
        "decreasing_steps": [right for left, right in zip(DOSES, DOSES[1:])
                             if rows[right]["projection"] < rows[left]["projection"] - .02],
        "full_response_pass": metrics(target, expected[100], outputs[0], outputs[100], mask)["passed"],
        "peak_projection": max(projection),
    }


def run(source, out):
    report = json.loads((source / "results.json").read_bytes())
    if report["assignments"] != ASSIGNMENTS or report["weight_sha256"] != CHECKPOINT:
        raise ValueError("Changed comparison scope")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed checkpoint")
    out.mkdir()
    results = {}
    for scene, donor in ASSIGNMENTS.items():
        target, authored = endpoints(source, report, scene)
        raw_ref, styled_ref = endpoints(source, report, donor)
        outputs, expected, hashes = {}, {}, {}
        board = Image.new("RGB", (1310, 480), "#222730")
        draw = ImageDraw.Draw(board)
        for col, amount in enumerate(DOSES):
            ref = finish_spatial(raw_ref, styled_ref, amount / 100)
            raw_grade = finish_spatial(target, authored, amount / 100)
            expected[amount] = finish_spatial(target, raw_grade, .8)
            image = finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
            if amount in (0, 100):
                role = "off" if amount == 0 else "film_b"
                sha = report["cases"][scene]["output_sha256"][role]
                prior = verified(source / f"{scene}-13-{role}.png", sha)
                np.testing.assert_array_equal(image, prior)
            path = out / f"{scene}-{amount:03}.png"
            Image.fromarray(image).save(path)
            outputs[amount], hashes[amount] = image, digest(path)
            for row, (label, pixels) in enumerate((("Reference", ref), ("13 (80%)", image), ("Expected", expected[amount]))):
                x, y = col * 262 + 4, row * 160
                draw.text((x, y + 4), f"{label} / {amount}%", fill="white")
                tile = Image.fromarray(pixels)
                tile.thumbnail((256, 140))
                board.paste(tile, (x, y + 20))
        all_pixels = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~all_pixels)["neutral_proxy"] for rgb in (target, authored)))
        if neutral.sum() < 256:
            raise ValueError("Insufficient neutral coverage")
        panel = out / f"{scene}-comparison.png"
        board.save(panel)
        results[scene] = {"donor": donor, "whole": response(target, expected, outputs, all_pixels),
                          "common_neutral": response(target, expected, outputs, neutral),
                          "outputs_sha256": hashes, "comparison_sha256": digest(panel)}
    json_to(out / "results.json", {"source_sha256": digest(source / "results.json"),
                                   "checkpoint_sha256": CHECKPOINT, "cases": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
