"""Complete the crossed Film B donor comparisons on three frozen real frames."""

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import digest, metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_hdm_hdr_display import json_to
from spatial_render import SpatialEngine, finish_spatial


def run(source, out):
    manifest = source / "results.json"
    report = json.loads(manifest.read_bytes())
    if report["assignments"] != ASSIGNMENTS or report["weight_sha256"] != CHECKPOINT:
        raise ValueError("Changed frozen source")
    images = {scene: endpoints(source, report, scene) for scene in ASSIGNMENTS}
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed checkpoint")
    out.mkdir()
    cases = {}
    for scene, donor in permutations(ASSIGNMENTS, 2):
        target, style = images[scene]
        expected = finish_spatial(target, style, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in images[scene]))
        if neutral.sum() < 256:
            raise ValueError(f"Insufficient neutral support: {scene}")
        preds, hashes = [], {}
        for role, guide in zip(("off", "film_b"), images[donor], strict=True):
            rgb = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
            if donor == ASSIGNMENTS[scene]:
                old = verified(source / f"{scene}-13-{role}.png", report["cases"][scene]["output_sha256"][role])
                np.testing.assert_array_equal(rgb, old)
            path = out / f"{scene}-from-{donor}-{role}.png"
            Image.fromarray(rgb).save(path)
            hashes[role] = digest(path)
            preds.append(rgb)
        cases[f"{scene}<-{donor}"] = {
            "target": scene, "reference": donor,
            "whole": metrics(target, expected, *preds, whole),
            "neutral": metrics(target, expected, *preds, neutral), "output_sha256": hashes,
        }
        board = Image.new("RGB", (1040, 170), "#222730")
        draw = ImageDraw.Draw(board)
        for col, (label, rgb) in enumerate(zip(("target", "13 / reference off", "13 / reference Film B", "expected"),
                                               (target, *preds, expected), strict=True)):
            tile = Image.fromarray(rgb)
            tile.thumbnail((250, 144))
            x = col * 260 + 4
            draw.text((x, 4), label, fill="white")
            board.paste(tile, (x, 20))
        board.save(out / f"{scene}-from-{donor}-comparison.png")
    patterns = {}
    for region in ("whole", "neutral"):
        patterns[region] = {
            side: [name for name in ASSIGNMENTS if
                   all(row[region]["projection"] < 0 for row in cases.values() if row[side] == name)]
            for side in ("target", "reference")
        }
    json_to(out / "results.json", {"source_sha256": digest(manifest), "checkpoint_sha256": CHECKPOINT,
                                   "cases": cases, "repeated_negative": patterns})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
