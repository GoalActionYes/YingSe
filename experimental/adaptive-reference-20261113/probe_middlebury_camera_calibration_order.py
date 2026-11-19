"""Check whether a fitted camera profile keeps a known creative edit intact."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_arri_film_b_dose import verified
from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
from probe_arri_two_looks import CHECKPOINT
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_middlebury_camera_channel_replay import CROSS_SHA, difference, regions
from probe_middlebury_camera_film_cross import PARENT_SHA
from spatial_render import SpatialEngine, finish_spatial


ORDERS = ("calibrate_first", "style_first")
TRAIN = np.array([(row + col) % 2 == 0 for row in range(4) for col in range(6)])


def patches(rgb):
    if rgb.shape != (260, 390, 3):
        raise ValueError("Unexpected camera chart layout")
    samples = [rgb[row * 65 + 16:row * 65 + 49, col * 65 + 16:col * 65 + 49].mean(axis=(0, 1))
               for row in range(4) for col in range(6)]
    return np.array(samples)


def profile(nikon, canon):
    selected = patches(nikon)[TRAIN]
    mat = np.column_stack((selected, np.ones(selected.shape[0])))
    coefficients, _, rank, _ = np.linalg.lstsq(mat, patches(canon)[TRAIN], rcond=None)
    if rank != 4:
        raise ValueError("Calibration fit lacks full rank")
    return coefficients


def calibrate(rgb, coefficients):
    unbounded = rgb @ coefficients[:3] + coefficients[3]
    clipped = ((unbounded < 0) | (unbounded > 255)).any(axis=2).mean()
    pixels = np.rint(np.clip(unbounded, 0, 255)).astype(np.uint8)
    return pixels, float(clipped)


def mae(left, right):
    return float(np.abs(left.astype(float) - right).mean())


def save(out, name, rgb):
    path = out / f"{name}.png"
    Image.fromarray(rgb).save(path)
    return digest(path)


def load_reference(base, record):
    actual = base / "middlebury-camera-assets-v1"
    crossed = base / "middlebury-camera-film-cross-v1"
    return {camera: {
        "off": verified(actual / f"{camera}.png", record["source"]["files"][camera]["sha256"]),
        "on": verified(crossed / f"{camera}-film.png", record["inputs"][camera]["edited_sha256"]),
    } for camera in ("canon", "nikon")}


@torch.inference_mode()
def run(base, out):
    crossed = base / "middlebury-camera-film-cross-v1"
    old = verified_json(crossed / "results.json", CROSS_SHA)
    parent = base / "arri-two-looks-v2"
    previous = verified_json(parent / "results.json", PARENT_SHA)
    refs = load_reference(base, old)
    canon, nikon = (refs[camera] for camera in ("canon", "nikon"))
    cube = base / "arri-shared-look-assets-v1/film-a.cube"
    if digest(cube) != LOOK_SHA or old["model_sha256"] != CHECKPOINT:
        raise ValueError("Changed frozen model or grade")
    view, inverse, film = transforms(cube)

    def style(rgb):
        log = converted(rgb.astype(np.float32) / 255, inverse)
        return quantize(np.clip(converted(converted(log, film), view), 0, 1))

    np.testing.assert_array_equal(style(nikon["off"]), nikon["on"])
    np.testing.assert_array_equal(style(canon["off"]), canon["on"])
    coeffs = profile(nikon["off"], canon["off"])
    common, clipped = calibrate(nikon["off"], coeffs)
    after_style, after_clipped = calibrate(nikon["on"], coeffs)
    prepared = {"off": common, "calibrate_first": style(common), "style_first": after_style}
    correct = patches(canon["off"])[~TRAIN]
    before = mae(patches(nikon["off"])[~TRAIN], correct)
    after = mae(patches(common)[~TRAIN], correct)
    qualified = after <= .5 * before and clipped <= .05
    out.mkdir()
    source = {
        "coefficients": coeffs.tolist(), "fit_cells": np.flatnonzero(TRAIN).tolist(),
        "holdout_cells": np.flatnonzero(~TRAIN).tolist(), "holdout_mae_before": before,
        "holdout_mae_after": after, "clipped_off": clipped, "clipped_style_first": after_clipped,
        "qualified": qualified, "order_difference": mae(prepared["calibrate_first"], after_style),
        "reference_diff_from_canon": {
            role: mae(guide, canon["off" if role == "off" else "on"]) for role, guide in prepared.items()
        },
        "sha256": {role: save(out, f"reference-{role}", guide) for role, guide in prepared.items()},
    }
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed loaded checkpoint")
    cases, gates = {}, dict.fromkeys(ORDERS, qualified)
    for scene, old_result in old["results"].items():
        files = previous["preparation"]["film_a"][scene]["images"]
        target, authored = (verified(parent / "film_a" / f"{scene}-{role}.png", files[role])
                            for role in ("off", "on"))
        expected = finish_spatial(target, authored, .8)
        masks = regions(scene, target, authored, old_result["regions"])
        outputs, identities = {}, {}
        for role, guide in prepared.items():
            outputs[role] = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
            identities[role] = save(out, f"{scene}-{role}", outputs[role])
        canon_target = {role: verified(crossed / f"{scene}-canon-{role}.png", old_result["hashes"]["canon"][role])
                        for role in ("off", "on")}
        rows = {}
        for label, mask in masks.items():
            arms = {}
            for role in ORDERS:
                arms[role] = {
                    "style": metrics(target, expected, outputs["off"], outputs[role], mask),
                    "camera_difference": {
                        "off": difference(outputs["off"][mask].astype(float) - canon_target["off"][mask]),
                        "on": difference(outputs[role][mask].astype(float) - canon_target["on"][mask]),
                    },
                }
            rows[label] = {
                "baseline": old_result["regions"][label], "arms": arms,
                "order_difference": difference(outputs["style_first"][mask].astype(float)
                                               - outputs["calibrate_first"][mask]),
            }
        for role in ORDERS:
            gates[role] &= all(rows[label]["arms"][role]["style"]["passed"] for label in ("whole", "neutral"))
            gates[role] &= all(side["rgb"] <= 3 for region in rows.values()
                               for side in region["arms"][role]["camera_difference"].values())
        cases[scene] = {"hashes": identities, "regions": rows}
    json_to(out / "results.json", {"parent_sha256": CROSS_SHA, "reference": source, "cases": cases,
                                   "joint_gates": gates, "passed": any(gates.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
