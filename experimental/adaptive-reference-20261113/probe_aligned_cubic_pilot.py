"""Test a fixed cubic color map behind the pinned same-frame checks."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from probe_frozen_candidate_development_crosscheck import BASE, CHECKPOINT, COUNTS, STRENGTH
from probe_frozen_same_content_control import build_cases, measure
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_same_content_app_path import cross_certification
from probe_target_mirror_chroma import checked_rgb, panel, save
from reference_color_app.aligned_reference import fit_aligned
from reference_color_app.contracts import Mapping
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial
from verify_development_inputs import verify


PARENT = "same-content-app-path-v1"
PARENT_SHA = "e17a15cbe8402153f34f113334f34e7c6de7cdb217ab15c05c0db74d3db89e77"
POWERS = sorted((powers for powers in product(range(4), repeat=3) if sum(powers) <= 3), key=sum)
RIDGE = 1e-5
FIT_FAILURES = {"Independent spatial holdout failed", "Local content or color mismatch"}


def design(rgb):
    scaled = 2 * rgb - 1
    return np.stack([scaled[..., 0] ** i * scaled[..., 1] ** j * scaled[..., 2] ** k
                     for i, j, k in POWERS], axis=-1)


def thumbnail_pair(target, reference):
    h, w = target.shape[:2]
    scale = min(1, 256 / max(h, w))
    size = max(16, round(w * scale)), max(16, round(h * scale))
    return [np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)) for rgb in (target, reference)]


def learn(a, b, train):
    a, b = (rgb.reshape(-1, 3).astype(np.float64) / 255 for rgb in (a, b))
    x = design(a)
    nonlinear = np.diag([sum(powers) >= 2 for powers in POWERS]).astype(float)
    weights = []
    for channel in range(3):
        usable = train & (b[:, channel] > 2 / 255) & (b[:, channel] < 253 / 255)
        u = x[usable]
        delta = (b[:, channel] - a[:, channel])[usable]
        regularizer = np.sqrt(RIDGE * u.shape[0]) * nonlinear
        fit, _, rank, _ = np.linalg.lstsq(np.vstack((u, regularizer)), np.r_[delta, np.zeros(len(POWERS))], rcond=None)
        if rank != len(POWERS) or not np.isfinite(fit).all():
            return None
        weights.append(fit)
    return np.stack(weights, axis=1)


def cubic_mapping(weights):
    axis = np.linspace(0, 1, 64)
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    graded = np.clip(rgb + design(rgb) @ weights, 0, 1)
    return Mapping(np.stack([.2 * rgb + .8 * graded, graded]).astype(np.float32), {})


def certify(target, reference):
    if np.array_equal(target, reference):
        return {"route": "identity"}, None
    affine = fit_aligned(target, reference)
    if affine["accepted"] or affine["reason"] not in FIT_FAILURES:
        return {"route": "affine" if affine["accepted"] else "original_fallback", "affine": affine}, None
    a, b = thumbnail_pair(target, reference)
    yy, xx = np.indices(a.shape[:2])
    train = (xx // 8 + yy // 8) % 2 == 0
    weights = learn(a, b, train.ravel())
    if weights is None:
        return {"route": "original_fallback", "affine": affine, "reason": "Unstable cubic fit"}, None
    mapping = cubic_mapping(weights)
    error = np.abs(apply_mapping(a, mapping, 1).astype(np.int16) - b)
    tile_errors = [float(error[j:j + 8, i:i + 8].mean())
                   for j in range(0, a.shape[0], 8) for i in range(0, a.shape[1], 8)]
    mae, p95, tile = float(error[~train].mean()), float(np.quantile(error[~train], .95)), max(tile_errors)
    accepted = mae <= 1.5 and p95 <= 4 and tile <= 6
    return {
        "route": "cubic" if accepted else "original_fallback", "affine": affine,
        "cubic": {"holdout_mae": mae, "holdout_p95": p95, "max_tile_mae": tile, "holdout_pixels": int((~train).sum())},
    }, mapping if accepted else None


def check_negatives(target):
    uniform = np.rint(.85 * target.astype(float) + 12).astype(np.uint8)
    local = target.copy()
    h, w = target.shape[:2]
    patch = local[h // 4:3 * h // 4, w // 4:3 * w // 4, 0]
    patch[:] = np.minimum(patch.astype(np.int16) + 64, 255)
    return {role: certify(target, reference)[0] for role, reference in
            (("identity", target), ("uniform", uniform), ("mirror", target[:, ::-1].copy()), ("local", local))}


def run(out):
    verify(BASE, COUNTS)
    original = verified_json(BASE / PARENT / "results.json", PARENT_SHA)
    if original["checkpoint_sha256"] != CHECKPOINT or original["strength"] != STRENGTH:
        raise ValueError("Frozen app controls changed")
    cases = list(build_cases())
    cross = cross_certification(cases, certify=lambda a, b: certify(a, b)[0])
    results = {"background": {}, "film_b": {}}
    out.mkdir()
    for suite, name, target, reference, expected, protected in cases:
        prefix = f"{suite}-{name}"
        archived = original["same_content"][suite][name]
        before = checked_rgb(BASE / PARENT / f"{prefix}-on.png", archived["states"]["on"]["output_sha256"])
        decision, mapping = certify(target, reference)
        if decision["affine"] != archived["states"]["on"]["certification"]:
            raise ValueError(f"Original app certification changed: {name}")
        output = before if mapping is None else finish_spatial(target, apply_mapping(target, mapping, 1), STRENGTH)
        scores = measure(suite, target, expected, {"off": target, "on": output}, protected)
        if mapping is None and scores != archived["metrics"]:
            raise ValueError(f"Original fallback metrics changed: {name}")
        results[suite][name] = {
            "decision": decision, "baseline": archived["metrics"]["from_target"], "candidate": scores,
            "baseline_protected_mae": archived["metrics"]["on_protected_mae_rgb8"],
            "negative_routes": check_negatives(target), "output_sha256": save(out / f"{prefix}.png", output),
            "panel_sha256": panel(out / f"{prefix}-comparison.png", {
                "matched reference": {"source": target, "reference": reference, "13": before,
                                      "cubic pilot": output, "known edit": expected}}),
        }
    counts = {suite: sum(all(region["passed"] for region in row["candidate"]["from_target"].values())
                         for row in block.values()) for suite, block in results.items()}
    negatives = [row["route"] == "cubic" for group in cross.values() for row in group.values()]
    regressions = []
    for block in results.values():
        for row in block.values():
            negatives.extend(record["route"] == "cubic" for record in row["negative_routes"].values())
            before = row["baseline_protected_mae"]
            if before is not None and row["decision"]["route"] == "cubic":
                regressions.append(row["candidate"]["on_protected_mae_rgb8"] - before)
    maximum_regression = max(regressions, default=0.)
    json_to(out / "results.json", {
        "parent_sha256": PARENT_SHA, "strength": STRENGTH, "cases": results, "cross_content": cross,
        "passing_cases": counts, "negative_cubic_activations": sum(negatives), "negative_cases": len(negatives),
        "max_protected_increase": maximum_regression,
        "passed": counts["background"] >= 3 and counts["film_b"] >= 2 and not any(negatives) and maximum_regression <= 2,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
