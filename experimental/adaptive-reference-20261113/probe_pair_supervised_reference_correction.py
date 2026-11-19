"""Learn a reference RGB correction from physically paired captures, retaining a look."""

import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest, metrics, quantize
from probe_arri_two_looks import CHECKPOINT
from probe_capture_invariant_look import natural_scenes, render_style, style_transforms
from probe_chromamill_capture_cross import LEVELS, LIGHTS, REVISION, SCENES, decode_rgb8, source_members
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_inretouch_registration import gradient, shifts
from probe_middlebury_camera_channel_replay import difference
from spatial_render import SpatialEngine, finish_spatial


TRAIN = ("Scene3_a", "Scene9_b", "Scene10_a", "Scene23_a")
CAPTURE_SHA = "ae4bdcfbaec10bf43c11b882138745ef02f86f58011d6102a43a666630e83789"
STATES = ("off", "on")


def features(rgb, conditioned, step=1):
    normalized = rgb.astype(float) / 255
    r, g, b = normalized[::step, ::step].reshape(-1, 3).T
    design = np.column_stack((np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b))
    if not conditioned:
        return design
    mean = normalized.mean(axis=(0, 1))
    return np.concatenate((design, *(design * value for value in mean)), axis=1)


def fit(data, conditioned):
    rows, output = [], []
    for photos in data.values():
        for (light, level), states in photos.items():
            for state, photo in states.items():
                ideal = photos["white", level][state]
                rows.append(features(photo, conditioned, step=8))
                output.append((ideal[::8, ::8].astype(float) - photo[::8, ::8]).reshape(-1, 3) / 255)
    x, y = np.concatenate(rows), np.concatenate(output)
    total = len(x)
    weights = np.linalg.solve(x.T @ x / total + .001 * np.eye(x.shape[1]), x.T @ y / total)
    return weights, total


def correct(rgb, weights, conditioned):
    adjusted = rgb / 255 + (features(rgb, conditioned) @ weights).reshape(rgb.shape)
    clipped = float(((adjusted < 0) | (adjusted > 1)).any(-1).mean())
    return quantize(adjusted), clipped


def load_training(folder, transform):
    import json

    receipt = json.loads((folder / "receipt.json").read_bytes())
    if receipt["revision"] != REVISION or {k: v["path"] for k, v in receipt["files"].items()} != source_members(TRAIN):
        raise ValueError("Changed training sources")
    data, reports = {}, {}
    for scene in TRAIN:
        images, shapes = {}, set()
        for light, level in product(LIGHTS, LEVELS):
            name = f"{scene}-{light}-{level}"
            path = folder / f"{name}.png"
            record = receipt["files"][name]
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError(f"Changed training pixels: {name}")
            rgb, format_info = decode_rgb8(path)
            shapes.add(tuple(format_info["shape"]))
            images[light, level] = {"off": rgb, "on": render_style(rgb, transform)}
        rows = {}
        if len(shapes) == 1:
            for level in LEVELS:
                white, blue = (images[hue, level]["off"] for hue in LIGHTS)
                rows[str(level)] = shifts(gradient(white), gradient(blue))[0]
        qualified = len(rows) == 2 and all(row["zero"] >= .60 and row["peak_gain"] <= .02 for row in rows.values())
        reports[scene] = {"alignment": rows, "qualified": qualified}
        if qualified:
            data[scene] = images
    return data, reports, receipt


def save(out, name, rgb):
    output = out / f"{name}.png"
    Image.fromarray(rgb).save(output)
    return digest(output)


def score_target(target, expected, images, baseline, masks):
    rows = {}
    for region, area in masks.items():
        effects, styles, white = {}, {}, {}
        for level in LEVELS:
            effects[str(level)] = {}
            for role in STATES:
                delta = (images["blue", level][role][area].astype(float)
                         - images["white", level][role][area])
                row = difference(delta)
                old = baseline["regions"][region]["factors"][role][f"color-{level}"]["rgb"]
                effects[str(level)][role] = row | {"old_rgb": old, "passed": row["rgb"] <= min(3, .5 * old)}
            for light in LIGHTS:
                styles[f"{light}-{level}"] = metrics(target, expected, *images[light, level].values(), area)
            for role in STATES:
                truth = expected if role == "on" else target
                white[f"{level}-{role}"] = float(np.abs(images["white", level][role][area].astype(float)
                                                           - truth[area]).mean())
        rows[region] = {"effects": effects, "styles": styles, "white_error": white}
    return rows


@torch.inference_mode()
def run(base, training, out):
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    look = style_transforms(base)["film_a"]
    data, quality, members = load_training(training, look)
    out.mkdir()
    if len(data) < 3:
        json_to(out / "results.json", {"training": quality, "source": members, "passed": False})
        return
    holdout = base / "chromamill-capture-cross-v1"
    parental = verified_json(holdout / "results.json", CAPTURE_SHA)

    def captures(scene, light, level, state):
        name = f"{light}-{level}"
        return verified(holdout / f"{scene}-{name}-{state}.png",
                        parental["prepared"][scene][name]["hashes"][state])

    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or parental["model_sha256"] != CHECKPOINT:
        raise ValueError("Changed frozen model")
    report = {}
    for arm, conditioned in (("shared", False), ("conditioned", True)):
        weights, count = fit(data, conditioned)
        folder = out / arm
        folder.mkdir()
        coefficients = folder / "coefficients.json"
        json_to(coefficients, {"coefficients": weights.tolist()})
        cases, negatives = {}, {}
        for scene, donor in (SCENES, SCENES[::-1]):
            prior = parental["cases"][scene]
            target = captures(scene, "white", 241, "off")
            photos, identities = {}, {}
            for light, level in product(LIGHTS, LEVELS):
                photos[light, level] = {}
                for state in STATES:
                    guide = captures(donor, light, level, state)
                    corrected, clipped = correct(guide, weights, conditioned)
                    name = f"{scene}-{light}-{level}-{state}"
                    grade = finish_spatial(target, engine.prepare(target, corrected).apply(target), .8)
                    identities[name] = {"reference": save(folder, f"{name}-reference", corrected),
                                        "output": save(folder, f"{name}-13", grade), "clipped": clipped}
                    photos[light, level][state] = grade

            authored = captures(scene, "white", 241, "on")
            expected = finish_spatial(target, authored, .8)
            whole = np.ones(target.shape[:2], bool)
            neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (target, authored)))
            if neutral.sum() != parental["checks"][scene]["neutral_pixels"]:
                raise ValueError("Changed neutral scoring support")
            masks = {"whole": whole, "neutral": neutral}
            rows = score_target(target, expected, photos, prior, masks)
            for label, area in masks.items():
                delta = {}
                for level, role in product(LEVELS, STATES):
                    name = f"white-{level}-{role}"
                    old = verified(holdout / f"{scene}-{name}-13.png", prior["hashes"][name])
                    truth = expected if role == "on" else target
                    original_error = float(np.abs(old[area].astype(float) - truth[area]).mean())
                    delta[f"{level}-{role}"] = rows[label]["white_error"][f"{level}-{role}"] - original_error
                rows[label]["white_error_increase"] = delta
            passed = all(item["passed"] for col in rows["neutral"]["effects"].values() for item in col.values())
            passed &= all(item["passed"] for row in rows.values() for item in row["styles"].values())
            passed &= all(value <= 2 for row in rows.values() for value in row["white_error_increase"].values())
            cases[scene] = {"rows": rows, "hashes": identities, "passed": passed}
        for scene, pictures in natural_scenes(base).items():
            for state in ("off", "film_b"):
                name = f"{scene}-{state}"
                photo = pictures[state]
                result, clipped = correct(photo, weights, conditioned)
                error = float(np.abs(result.astype(float) - photo).mean())
                negatives[name] = {"rgb_mae": error, "clipped": clipped, "sha256": save(folder, name, result),
                                   "passed": error <= 1.5}
        report[arm] = {"cases": cases, "negatives": negatives, "train_rows": count, "weights_sha256": digest(coefficients),
                       "passed": all(row["passed"] for row in (*cases.values(), *negatives.values()))}
    json_to(out / "results.json", {"training": quality, "source": members, "parent_sha256": CAPTURE_SHA,
                                   "arms": report, "passed": report["conditioned"]["passed"] and not report["shared"]["passed"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "training", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
