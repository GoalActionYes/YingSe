"""Use an offline paired lamp calibration before applying the known digital look."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
from probe_arri_sdr_renderability import transforms
from probe_hdm_chart import CHECKPOINT
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import encode, linear_rgb, verified_json
from probe_mls_film_light_cross import SOURCE_SHA
from spatial_render import SpatialEngine, finish_spatial


def calibrate(standard, blue):
    support = np.logical_and.reduce([(rgb.min(-1) >= 16) & (rgb.max(-1) <= 239) for rgb in (standard, blue)])
    if support.sum() < 256:
        raise ValueError("Insufficient paired lamp coverage")
    a, b = map(linear_rgb, (standard, blue))
    gain = np.median(a[support] / b[support], axis=0)
    corrected = b * gain
    return encode(corrected), {"support_pixels": int(support.sum()), "gain": gain.tolist(),
                               "out_of_gamut_pixel_fraction": float((corrected.max(-1) > 1).mean())}


def save(path, image):
    Image.fromarray(image).save(path)
    return digest(path)


def run(base, out):
    source = base / "mls-linear-pilot"
    prepared = verified_json(source / "results.json", SOURCE_SHA)
    prior = base / "mls-film-light-cross-v1"
    old = json.loads((prior / "results.json").read_bytes())
    lut = base / "arri-shared-look-assets-v1/film-a.cube"
    if old["source_sha256"] != SOURCE_SHA or old["weight_sha256"] != CHECKPOINT or digest(lut) != LOOK_SHA:
        raise ValueError("Changed frozen baseline")
    view, inverse, film = transforms(lut)
    out.mkdir()
    scenes = {}
    controls = {}
    for scene, row in prepared["scenes"].items():
        filenames = [f"{scene}_2HAL_DESK{suffix}.png" for suffix in ("", "_LED-B025")]
        standard, blue = (verified(source / name, row["inputs_sha256"][name]) for name in filenames)
        authored = verified(prior / f"{scene}-standard-reference-film.png", old["references"][scene]["standard"])
        corrected, gain = calibrate(standard, blue)
        log = converted(corrected.astype(np.float32) / 255, inverse)
        edited = quantize(np.clip(converted(converted(log, film), view), 0, 1))
        scenes[scene] = {"target": standard, "expected": authored, "off": corrected, "on": edited}
        controls[scene] = gain | {
            "reference_hashes": {role: save(out / f"{scene}-reference-{role}.png", img)
                                 for role, img in (("off", corrected), ("on", edited))},
            "mean_reference_error_before": float(np.abs(blue.astype(float) - standard).mean()),
            "mean_reference_error_after": float(np.abs(corrected.astype(float) - standard).mean()),
        }

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 checkpoint")
    results = {}
    for scene, spec in old["cases"].items():
        donor = spec["reference_scene"]
        target, authored = (scenes[scene][key] for key in ("target", "expected"))
        expected = finish_spatial(target, authored, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (target, authored)))
        if neutral.sum() != spec["neutral_pixels"]:
            raise ValueError("Changed scored neutral region")
        predictions, spill, hashes = {}, {}, {}
        for role in ("off", "on"):
            guide = scenes[donor][role]
            rgb = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
            hashes[role] = save(out / f"{scene}-13-{role}.png", rgb)
            predictions[role] = rgb
            previous = verified(prior / f"{scene}-from-{donor}-standard-{role}.png",
                                spec["outputs_sha256"]["standard"][role])
            spill[role] = float(np.abs(rgb[neutral].astype(float) - previous[neutral]).mean())
        response = {region: metrics(target, expected, predictions["off"], predictions["on"], mask)
                    for region, mask in (("whole", whole), ("neutral", neutral))}
        corrected_pass = max(spill.values()) <= 3 and all(row["passed"] for row in response.values())
        results[scene] = {"donor": donor, "response": response, "neutral_lamp_spill_mae": spill,
                          "original_lamp_spill_mae": spec["neutral_lamp_spill_mae"], "outputs_sha256": hashes,
                          "corrected_arm_passed": corrected_pass,
                          "passed": corrected_pass and all(row["passed"] for row in spec["response"]["standard"].values())}
    json_to(out / "results.json", {"baseline_sha256": digest(prior / "results.json"), "controls": controls,
                                   "cases": results, "all_passed": all(row["passed"] for row in results.values())})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("out", type=Path)
    run(**vars(cli.parse_args()))
