"""Check whether an illuminant estimate can safely trigger a frozen reference correction."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest, quantize
from probe_arri_two_looks import CHECKPOINT
from probe_capture_invariant_look import natural_scenes
from probe_cti_reference_screen import ASSET_SHA, estimate, model
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_pair_supervised_reference_correction import CAPTURE_SHA, LEVELS, LIGHTS, SCENES, STATES, correct, save, score_target
from spatial_render import SpatialEngine, finish_spatial


ESTIMATES_SHA = "f010a01952d7269cdfd36d91be08aad69d90756eff844416f5ff6fbed60987a3"
MODEL_SHA = "6a0a7d627d8e33e347387fd8563012ec01d60c8bb96b612f66186fc500cb36bc"


def linear(rgb):
    encoded = rgb.astype(float) / 255
    return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)


def tint(rgb, gains):
    values = linear(rgb) * gains
    clipped = float((values.max(axis=2) > 1).mean())
    values = np.clip(values, 0, 1)
    return quantize(np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)), clipped


def masks_for(base, styled):
    whole = np.ones(base.shape[:2], bool)
    return {"whole": whole, "neutral": np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"]
                                                        for rgb in (base, styled)))}


def mae(a, b):
    return float(np.abs(a.astype(float) - b).mean())


@torch.inference_mode()
def run(base, assets, out):
    source = base / "chromamill-capture-cross-v1"
    chosen = base / "pair-supervised-reference-correction-v1"
    captures = verified_json(source / "results.json", CAPTURE_SHA)
    trained = verified_json(chosen / "results.json", MODEL_SHA)["arms"]["conditioned"]
    estimates = verified_json(base / "cti-reference-screen-v1/results.json", ESTIMATES_SHA)
    if estimates["assets"] != ASSET_SHA or estimates["model_sha256"] != CHECKPOINT:
        raise ValueError("Changed model evidence")
    params = verified_json(chosen / "conditioned/coefficients.json", trained["weights_sha256"])
    weights = np.array(params["coefficients"])
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 checkpoint")
    net = model(assets, engine.device)
    out.mkdir()

    def image(scene, light, level, state):
        tag = f"{light}-{level}"
        return verified(source / f"{scene}-{tag}-{state}.png", captures["prepared"][scene][tag]["hashes"][state])

    physical, digital = {}, {}
    for target_scene, donor in (SCENES, SCENES[::-1]):
        target = image(target_scene, "white", 241, "off")
        authored = image(target_scene, "white", 241, "on")
        expected = finish_spatial(target, authored, .8)
        masks = masks_for(target, authored)
        parent_case = captures["cases"][target_scene]
        views, selected = {}, {}
        for light, level in product(LIGHTS, LEVELS):
            views[light, level] = {}
            for state in STATES:
                tag = f"{light}-{level}-{state}"
                applied = estimates["references"][f"{donor}-{tag}"]["nearest"] == "blue"
                if applied:
                    row = trained["cases"][target_scene]["hashes"][f"{target_scene}-{tag}"]
                    path = chosen / "conditioned" / f"{target_scene}-{tag}-13.png"
                    views[light, level][state] = verified(path, row["output"])
                else:
                    path = source / f"{target_scene}-{tag}-13.png"
                    views[light, level][state] = verified(path, parent_case["hashes"][tag])
                selected[tag] = applied
        scores = score_target(target, expected, views, parent_case, masks)
        joint = all(row["passed"] for level in scores["neutral"]["effects"].values() for row in level.values())
        joint &= all(row["passed"] for region in scores.values() for row in region["styles"].values())
        physical[target_scene] = {"applied": selected, "scores": scores, "passed": joint}

        records = {}
        for level in LEVELS:
            mean_white = linear(image(donor, "white", level, "off")).mean((0, 1))
            mean_blue = linear(image(donor, "blue", level, "off")).mean((0, 1))
            if (mean_white <= 0).any():
                raise ValueError("No support for matched digital tint")
            gains = mean_blue / mean_white
            digitally_tinted, clip = tint(image(donor, "white", level, "off"), gains)
            mean_difference = np.abs(linear(digitally_tinted).mean(axis=(0, 1)) - mean_blue)
            qualified = bool((mean_difference <= .02).all() and clip <= .05)
            rows = {}
            for state in STATES:
                reference, clipped = tint(image(donor, "white", level, state), gains)
                tag = f"{target_scene}-{level}-{state}"
                _, light = estimate(net, reference, engine.device)
                applied = light["nearest"] == "blue"
                original = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
                if applied:
                    guide, _ = correct(reference, weights, True)
                    result = finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                else:
                    result = original
                truth, _ = tint(target if state == "off" else authored, gains)
                truth = finish_spatial(target, truth, .8)
                errors = {name: {"baseline": mae(original[mask], truth[mask]), "gated": mae(result[mask], truth[mask])}
                          for name, mask in masks.items()}
                degraded = any(item["gated"] - item["baseline"] > 2 for item in errors.values())
                sources = {name: save(out, f"{tag}-{name}", item)
                           for name, item in (("reference", reference), ("original", original), ("gated", result))}
                rows[state] = {"estimate": light, "applied": applied, "clipped": clipped, "scores": errors,
                               "degraded": degraded, "hashes": sources,
                               "physical_vs_digital_rgb": mae(image(donor, "blue", level, state), reference)}
            records[str(level)] = {"qualified": qualified, "gains": gains.tolist(), "linear_mean_difference": mean_difference.tolist(),
                                   "states": rows, "passed": qualified and all(not row["applied"] and not row["degraded"]
                                                                               for row in rows.values())}
        digital[target_scene] = records
    negatives = {}
    for scene, photos in natural_scenes(base).items():
        for state in ("off", "film_b"):
            tag = f"{scene}-{state}"
            label = estimates["references"][tag]["nearest"]
            reference = photos[state]
            result = correct(reference, weights, True)[0] if label == "blue" else reference
            negatives[tag] = {"nearest": label, "change_rgb_mae": mae(reference, result), "passed": label != "blue"}
    gates = {"physical": all(row["passed"] for row in physical.values()),
             "digital": all(row["passed"] for scene in digital.values() for row in scene.values()),
             "ordinary": all(row["passed"] for row in negatives.values())}
    json_to(out / "results.json", {"parents": {"capture": CAPTURE_SHA, "estimates": ESTIMATES_SHA, "correction": MODEL_SHA},
                                   "physical": physical, "digital": digital, "natural": negatives,
                                   "gates": gates, "passed": all(gates.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "assets", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
