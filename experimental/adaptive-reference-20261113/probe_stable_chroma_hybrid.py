"""Test existing global chroma at the original spatial output's luminance."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from probe_hdm_chart import REF_SCENE, REF_SHA, chart_metrics, get_reference
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_target_mirror_chroma import (
    BASE, ROOT, CHECKPOINT, LUMA_INTEGER, STRENGTH, background, checked_rgb, facd, panel, save,
)
from reference_color_app.features import extract_features
from reference_color_app.lut import apply_mapping, build_mapping
from reference_color_app.pipeline import algorithm_fingerprint
from spatial_render import SpatialEngine


def candidate(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True, "max_luma_error": 0.}
    mapping = build_mapping(extract_features(target, 0), extract_features(reference, 1))
    global_rgb = apply_mapping(target, mapping, STRENGTH)
    result, limited = render(chroma(global_rgb, global_rgb @ LUMA), baseline @ LUMA)
    error = np.abs((result.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000
    return result, {"max_luma_error": float(error), "gamut_limited_fraction": limited}


def chart(engine, out, grade=candidate, label="hybrid"):
    result = BASE / "hdm-chart-pilot-v1"
    report = json.loads((result / "results.json").read_bytes())
    prep = verified_json(BASE / "hdm-chart-prepared-v1" / "prepared.json", report["prepared_sha256"])
    rois = json.loads((ROOT / "experimental/adaptive-reference-20261113/hdm-chart-pilot-regions.json").read_bytes())
    ref_folder = BASE / "arri-shared-look-cross-source-v1"
    guides = verified_json(ref_folder / "results.json", REF_SHA)["prepared"][REF_SCENE]["images"]
    reference = {role: get_reference(ref_folder, role, guides[role]) for role in ("off", "on")}
    original = {role: checked_rgb(result / f"{role}.png", sha)
                for role, sha in report["outputs_sha256"].items()}
    full = checked_rgb(BASE / "hdm-chart-prepared-v1/hdr_testimage-on.png", prep["chart"]["images"]["on"])
    target = original["target"]
    valid = ((target.min(-1) > 0) & (target.max(-1) < 255)
             & (full.min(-1) > 0) & (full.max(-1) < 255))
    candidate_images = dict(original)
    out.mkdir()
    outputs = {}
    for role, ref in reference.items():
        key = f"13-{role}"
        result_rgb, audit = grade(engine, target, ref, original[key])
        candidate_images[key] = result_rgb
        outputs[role] = dict(audit, sha256=save(out / f"{role}.png", result_rgb))
    metrics = {}
    for name, polygon in rois["regions"].items():
        raster = Image.new("L", (512, 288))
        ImageDraw.Draw(raster).polygon([tuple(pair) for pair in polygon], fill=255)
        mask = valid & (np.asarray(raster) > 0)
        if chart_metrics(original, mask) != report["metrics"][name]:
            raise ValueError(f"Original chart scores changed: {name}")
        metrics[name] = chart_metrics(candidate_images, mask)
    response_pass = sum(row["passed"] for row in metrics.values())
    max_off = max(row["off_chroma_mae"] for row in metrics.values())
    comparison = panel(out / "comparison.png", {
        role: {"target": target, "reference": ref, "13": original[f"13-{role}"],
               label: candidate_images[f"13-{role}"],
               "expected": original["expected"] if role == "on" else target}
        for role, ref in reference.items()
    })
    return {"baseline": report["metrics"], "candidate": metrics, "outputs": outputs,
            "response_pass": response_pass, "max_off": max_off,
            "passed": response_pass >= 5 and max_off <= 3, "comparison_sha256": comparison}


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    sample = np.array([[[0, 40, 80]]], dtype=np.uint8)
    np.testing.assert_array_equal(candidate(engine, sample, sample.copy(), 255-sample)[0], sample)
    out.mkdir()
    charts = chart(engine, out / "chart")
    bg = background(engine, out / "bg-real", candidate, "global chroma")
    authored = facd(engine, out / "facd", candidate, "global chroma")
    samples = [charts, *bg["cases"].values(), *authored["cases"].values()]
    maximum = max(stat["max_luma_error"] for site in samples for stat in site["outputs"].values())
    json_to(out / "results.json", {
        "checkpoint_sha256": CHECKPOINT, "strength": STRENGTH, "stable_fingerprint": algorithm_fingerprint(),
        "chart": charts, "background": bg, "facd": authored, "max_luma_error": maximum,
        "passed": charts["passed"] and bg["passed"] and authored["passed"] and maximum <= .5,
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
