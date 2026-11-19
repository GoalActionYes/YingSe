"""Compare early and late clipping on the same frozen expanded grids."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_frozen_candidate_development_crosscheck import (
    BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256, background, film_b,
)
from probe_frozen_render_blend_bound import FULL, FULL_SHA
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from verify_development_inputs import verify


def pixel_sha(rgb):
    return sha256(rgb.tobytes()).hexdigest()


def render(engine, target, reference, baseline):
    grid = appearance_grid(engine.model, engine.api, engine.analysis(target), engine.analysis(reference))
    prepared = PreparedSpatial(engine, grid, {})
    early = finish_spatial(target, prepared.apply(target), STRENGTH)
    raw = prepared.float_output(target, chunk_rows=256)
    mixed = target.astype(np.float32) / 255 * (1 - STRENGTH) + raw * STRENGTH
    late = np.rint(np.clip(mixed, 0, 1) * 255).astype(np.uint8)
    early_clip = (raw < 0) | (raw > 1)
    late_clip = (mixed < 0) | (mixed > 1)
    diff = late.astype(np.int16) - early
    return late, {
        "control_pixels_sha256": pixel_sha(early),
        "raw_clipped_channel_fraction": float(early_clip.mean()),
        "mixed_clipped_channel_fraction": float(late_clip.mean()),
        "changed_channel_fraction": float(np.count_nonzero(diff) / diff.size),
        "late_vs_control_mae_rgb8": float(np.abs(diff).mean()),
        "late_vs_13_mae_rgb8": float(np.abs(late.astype(np.int16) - baseline).mean()),
    }


def verify_control(suites, parent):
    for suite, results in suites.items():
        for key, row in results["cases"].items():
            frozen = parent["suites"][suite]["cases"][key]
            for state, output in row["outputs"].items():
                if suite == "background":
                    name = f"{key}/{state}.png"
                else:
                    target, donor = key.split("<-")
                    name = f"{target}-from-{donor}-{state}.png"
                archived = checked_rgb(BASE / FULL / suite / name, frozen["outputs"][state]["sha256"])
                if output["control_pixels_sha256"] != pixel_sha(archived):
                    raise ValueError(f"Frozen control pixels changed: {name}")


@torch.inference_mode()
def run(out):
    verify(BASE, COUNTS)
    parent = verified_json(BASE / FULL / "results.json", FULL_SHA)
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError("Expanded checkpoint changed")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    suites = {
        "background": background(engine, out / "background", render, "late clipping"),
        "film_b": film_b(engine, out / "film_b", render, "late clipping"),
    }
    verify_control(suites, parent)
    json_to(out / "results.json", {
        "parent_sha256": FULL_SHA, "checkpoint_sha256": WEIGHT_SHA256, "strength": STRENGTH,
        "suites": suites, "passed": all(row["passed"] for row in suites.values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
