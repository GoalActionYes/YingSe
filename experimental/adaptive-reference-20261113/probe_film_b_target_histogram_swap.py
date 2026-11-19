"""Swap only the target histogram in the frozen Film B comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import digest, metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_hdm_hdr_display import json_to
from probe_target_chroma_histogram import prepare
from spatial_render import SpatialEngine, finish_spatial


def run(source, crossed, out):
    upstream = source / "results.json"
    reference_run = crossed / "results.json"
    report = json.loads(upstream.read_bytes())
    baseline = json.loads(reference_run.read_bytes())
    if baseline["source_sha256"] != digest(upstream) or len(baseline["cases"]) != 6:
        raise ValueError("Changed crossed Film B comparison")
    images = {scene: endpoints(source, report, scene) for scene in ASSIGNMENTS}
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or baseline["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("Changed checkpoint")
    out.mkdir()
    cases = {}
    for name, row in baseline["cases"].items():
        scene, donor = row["target"], row["reference"]
        target, styled = images[scene]
        expected = finish_spatial(target, styled, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in images[scene]))
        scores = {}
        prefix = f"{scene}-from-{donor}"
        board = Image.new("RGB", (780, 510), "#222730")
        draw = ImageDraw.Draw(board)
        for index, sample in enumerate((scene, *(other for other in images if other != scene))):
            candidate, hashes = [], {}
            y = index * 170
            draw.text((4, y + 5), f"Statistics: {sample}", fill="white")
            for col, (role, guide) in enumerate(zip(("off", "film_b"), images[donor], strict=True), 1):
                plan = prepare(engine, target, guide, neutral_target=False, histogram_image=images[sample][0])
                result = finish_spatial(target, plan.apply(target), .8)
                if sample == scene:
                    old = verified(crossed / f"{prefix}-{role}.png", row["output_sha256"][role])
                    np.testing.assert_array_equal(old, result)
                path = out / f"{prefix}-stats-{sample}-{role}.png"
                image = Image.fromarray(result)
                image.save(path)
                image.thumbnail((250, 140))
                board.paste(image, (col * 260 + 4, y + 22))
                draw.text((col * 260 + 4, y + 5), role, fill="white")
                candidate.append(result)
                hashes[role] = digest(path)
            scores[sample] = {"whole": metrics(target, expected, *candidate, whole),
                              "neutral": metrics(target, expected, *candidate, neutral), "output_sha256": hashes}
        panel = out / f"{prefix}-comparison.png"
        board.save(panel)
        cases[name] = {"target": scene, "reference": donor, "statistics": scores,
                       "comparison_sha256": digest(panel)}
    json_to(out / "results.json", {"checkpoint_sha256": CHECKPOINT, "baseline_sha256": digest(reference_run),
                                   "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "crossed", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
