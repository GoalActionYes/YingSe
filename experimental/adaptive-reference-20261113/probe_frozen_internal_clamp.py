"""Test bounded monotone continuation of the frozen internal tone curves."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from internal_affine_stage import internal_affine
from probe_arri_film_b_dose import endpoints
from probe_frozen_candidate_development_crosscheck import (
    BASE, CHECKPOINT, COUNTS, STRENGTH, WEIGHT, WEIGHT_SHA256, background, film_b,
)
from probe_frozen_render_blend_bound import FULL, FULL_SHA
from probe_frozen_render_clip_order import pixel_sha, verify_control
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import checked_rgb
from spatial_render import SpatialEngine, coordinates, finish_spatial
from verify_development_inputs import manifest, verify

LUMA = np.array([.299, .587, .114])


def end_slopes(guide, grid):
    size = guide.lut_L.numel()
    if size < 2 or any(table.numel() != size for table in (guide.lut_Ct, guide.lut_Cp)):
        raise ValueError("Unexpected guide lengths")
    step = 2 / (size - 1)
    coords = grid.new_tensor([-1, -1 + step, 1 - step, 1]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    values = guide(coords)
    return (values[..., 1:2] - values[..., :1]) / step, (values[..., 3:4] - values[..., 2:3]) / step


def byte_rgb(value):
    return value[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def inspect_render(engine, target, reference):
    model = engine.model
    grid = appearance_grid(model, engine.api, engine.analysis(target), engine.analysis(reference))
    low_slope, high_slope = end_slopes(model.post_guide, grid)
    early, continuation = np.empty_like(target), np.empty_like(target)
    sides = np.empty(target.shape, dtype=np.int8)
    height, width = target.shape[:2]
    weights = F.softmax(model.sampler.L_weights, dim=1)
    for start in range(0, height, 256):
        stop = min(start + 256, height)
        content = engine.tensor(target[start:stop])
        guide = (model.sampler.pre_guide(content) * weights).sum(1)
        coords = coordinates(width, height, start, stop, guide)
        coeffs = F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=False).squeeze(2)
        affine = internal_affine(model, coeffs, content)
        native = model.post_guide(affine.clamp(-1, 1))
        if not torch.equal(native, model.apply_coeffs(coeffs, content)):
            raise ValueError("Internal affine reconstruction changed the control")
        continued = native + (affine + 1).clamp_max(0) * low_slope + (affine - 1).clamp_min(0) * high_slope
        sides[start:stop] = ((affine > 1).to(torch.int8) - (affine < -1).to(torch.int8))[0].permute(1, 2, 0).cpu().numpy()
        early[start:stop] = byte_rgb(native)
        continuation[start:stop] = byte_rgb(continued)
    return finish_spatial(target, early, STRENGTH), finish_spatial(target, continuation, STRENGTH), sides


def limits(target, current, sides):
    minimum = finish_spatial(target, np.zeros_like(target), STRENGTH)
    maximum = finish_spatial(target, np.full_like(target, 255), STRENGTH)
    return np.where(sides < 0, minimum, current), np.where(sides > 0, maximum, current)


def upper(off, on, desired, region, *, luma=False):
    off_min, off_max = off
    on_min, on_max = on
    delta_min = on_min.astype(np.int16) - off_max
    delta_max = on_max.astype(np.int16) - off_min
    if luma:
        delta_min, delta_max = delta_min @ LUMA, delta_max @ LUMA
    signal = desired[region]
    return float(np.where(signal >= 0, delta_max[region], delta_min[region]).ravel() @ signal.ravel()
                 / np.square(signal).sum())


def add_bounds(rows, intervals):
    for key, row in rows["background"]["cases"].items():
        location = BASE / "bg-real-editor-look" / key
        files = manifest(BASE, "bg-real-editor-look")["cases"][key]["artifacts"]
        target, expected = (checked_rgb(location / name, files[name]) for name in ("target.png", "expected-on.png"))
        signal = (expected.astype(float) - target) @ LUMA
        row["monotone_luma_upper"] = {
            variant: upper(*(intervals[row["outputs"][f"{variant}-{state}"]["interval"]]
                             for state in ("off", "on")), signal, np.ones(signal.shape, dtype=bool), luma=True)
            for variant in ("C1", "F1")
        }
    preparation = manifest(BASE, "arri-two-looks-v2")
    for key, row in rows["film_b"]["cases"].items():
        target, _ = key.split("<-")
        original, styled = endpoints(BASE / "arri-two-looks-v2", preparation, target)
        signal = finish_spatial(original, styled, STRENGTH).astype(float) - original
        row["monotone_whole_upper"] = upper(*(intervals[row["outputs"][state]["interval"]]
                                             for state in ("off", "film_b")), signal,
                                           np.ones(original.shape[:2], dtype=bool))


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
    intervals = []

    def compare(engine, target, reference, _baseline):
        old, new, sides = inspect_render(engine, target, reference)
        lo, hi = limits(target, old, sides)
        if np.any(new < lo) or np.any(new > hi):
            raise ValueError("Continuation falls outside its monotone bounds")
        index = len(intervals)
        intervals.append((lo, hi))
        difference = np.abs(new.astype(np.int16) - old)
        return new, {
            "interval": index, "control_pixels_sha256": pixel_sha(old),
            "below_fraction": float((sides < 0).mean()), "above_fraction": float((sides > 0).mean()),
            "changed_fraction": float(np.count_nonzero(difference) / difference.size),
            "continuation_change_mae_rgb8": float(difference.mean()),
            "continuation_change_max_rgb8": int(difference.max()),
        }

    out.mkdir()
    suites = {
        "background": background(engine, out / "background", compare, "internal curve continuation"),
        "film_b": film_b(engine, out / "film_b", compare, "internal curve continuation"),
    }
    verify_control(suites, parent)
    add_bounds(suites, intervals)
    json_to(out / "results.json", {
        "parent_sha256": FULL_SHA, "checkpoint_sha256": WEIGHT_SHA256, "strength": STRENGTH,
        "suites": suites, "passed": all(row["passed"] for row in suites.values()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
