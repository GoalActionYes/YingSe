"""Isolate reference-camera lightness from chroma on the fixed Film A cases."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import target_regions
from histogram_replay import conditions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest, metrics
from probe_arri_two_looks import CHECKPOINT, chart_masks
from probe_bg_real_channel_replay import histogram, infer
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_hdm_poker_material_roi import REGIONS
from probe_middlebury_camera_film_cross import PARENT_SHA
from probe_psbench_background_desaturation import REC709
from spatial_render import ROOT, SpatialEngine, finish_spatial


CROSS_SHA = "4088e7ebcc2c4cb995be2a5a3312c35c3431d4dd444dfa1ce79a5dceabaa43bc"
CAMERAS = ("canon", "nikon")
STATES = ("off", "on")


def regions(scene, target, authored, prior):
    whole = np.ones(target.shape[:2], dtype=bool)
    masks = {"whole": whole, "neutral": np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"]
                                                         for rgb in (target, authored)))}
    if scene == "hdr_testimage":
        masks.update({key: area for key, area in chart_masks(ROOT, {"off": target, "on": authored}).items()
                      if key.startswith("skin_")})
    else:
        for key in ("cloth_left", "cloth_right"):
            patch = Image.new("L", whole.shape[::-1])
            ImageDraw.Draw(patch).polygon(REGIONS[key], fill=255)
            masks[key] = np.asarray(patch) != 0
    if {key: int(mask.sum()) for key, mask in masks.items()} != {key: row["pixels"] for key, row in prior.items()}:
        raise ValueError("Frozen scoring region changed")
    return masks


def difference(delta):
    light = delta @ REC709
    return {"rgb": float(np.abs(delta).mean()), "luma": float(np.abs(light).mean()),
            "chroma": float(np.abs(delta - light[:, None]).mean())}


def evaluate(original, expected, outputs, masks):
    report = {}
    for label, area in masks.items():
        report[label] = {}
        for role in STATES:
            cc, cn, nc, nn = (outputs[pair][role][area].astype(float) for pair in product(CAMERAS, repeat=2))
            report[label][role] = {
                "baseline": difference(nn - cc),
                "same_luma": {"canon": difference(cn - cc), "nikon": difference(nn - nc)},
                "same_chroma": {"canon": difference(nc - cc), "nikon": difference(nn - cn)},
                "interaction": difference(nn - nc - cn + cc),
            }
        report[label]["style"] = {
            f"{light}_{color}": metrics(original, expected, *arms.values(), area)
            for (light, color), arms in outputs.items()
        }
    return report


@torch.inference_mode()
def run(base, out):
    cross = base / "middlebury-camera-film-cross-v1"
    source = base / "middlebury-camera-assets-v1"
    parent = base / "arri-two-looks-v2"
    previous = verified_json(cross / "results.json", CROSS_SHA)
    authored_targets = verified_json(parent / "results.json", PARENT_SHA)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or previous["model_sha256"] != CHECKPOINT:
        raise ValueError("Changed 13 checkpoint")
    guides = {
        camera: {
            "off": verified(source / f"{camera}.png", previous["source"]["files"][camera]["sha256"]),
            "on": verified(cross / f"{camera}-film.png", previous["inputs"][camera]["edited_sha256"]),
        } for camera in CAMERAS
    }
    if not all(previous["inputs"][camera]["profile"] == "untagged assumed sRGB" for camera in CAMERAS):
        raise ValueError("Frozen reference expected untagged sRGB")

    histograms = {}
    for role in STATES:
        canon, nikon = (histogram(engine, guides[camera][role]).cpu().numpy() for camera in CAMERAS)
        exchanged = conditions(canon, nikon)
        cc, cn, nc, nn = canon, exchanged["chroma_only"], exchanged["y_only"], exchanged["full"]
        np.testing.assert_array_equal(cc[:, :1], cn[:, :1])
        np.testing.assert_array_equal(nn[:, :1], nc[:, :1])
        np.testing.assert_array_equal(cc[:, 1:], nc[:, 1:])
        np.testing.assert_array_equal(nn[:, 1:], cn[:, 1:])
        histograms[role] = {key: torch.from_numpy(value).to(engine.device)
                            for key, value in zip(product(CAMERAS, repeat=2), (cc, cn, nc, nn), strict=True)}

    out.mkdir()
    cases, dominance, gates = {}, [], dict.fromkeys(CAMERAS, True)
    for scene, row in previous["results"].items():
        files = authored_targets["preparation"]["film_a"][scene]["images"]
        target, authored = (verified(parent / "film_a" / f"{scene}-{role}.png", files[role]) for role in STATES)
        content = engine.analysis(target)
        low = engine.api["TF"].resize(content, (64, 64), antialias=True)
        target_stats = histogram(engine, target)
        outputs, hashes = {}, {}
        for light, color in product(CAMERAS, repeat=2):
            key = (light, color)
            outputs[key], hashes[f"{light}_{color}"] = {}, {}
            for role in STATES:
                grade = infer(engine, target, low, target_stats, histograms[role][key])
                if light == color:
                    frozen = verified(cross / f"{scene}-{light}-{role}.png", row["hashes"][light][role])
                    np.testing.assert_array_equal(grade, frozen)
                else:
                    path = out / f"{scene}-{light}-y-{color}-chroma-{role}.png"
                    Image.fromarray(grade).save(path)
                    hashes[f"{light}_{color}"][role] = digest(path)
                outputs[key][role] = grade
        masks = regions(scene, target, authored, row["regions"])
        scores = evaluate(target, finish_spatial(target, authored, .8), outputs, masks)
        for camera in CAMERAS:
            for role in STATES:
                test = scores["neutral"][role]
                dominance.append(test["same_luma"][camera]["rgb"] <= .5 * test["baseline"]["rgb"])
            gates[camera] &= all(reg[side]["same_luma"][camera]["rgb"] <= 3
                                 for reg in scores.values() for side in STATES)
            gates[camera] &= all(scores[name]["style"][f"{camera}_{color}"]["passed"]
                                 for name in ("whole", "neutral") for color in CAMERAS)
        cases[scene] = {"regions": scores, "new_hashes": {key: value for key, value in hashes.items() if value}}
    json_to(out / "results.json", {"parent_sha256": CROSS_SHA, "model_sha256": CHECKPOINT, "cases": cases,
                                   "neutral_rgb_reduction_half": sum(dominance), "neutral_tests": len(dominance),
                                   "same_luma_joint_gates": gates, "passed": any(gates.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
