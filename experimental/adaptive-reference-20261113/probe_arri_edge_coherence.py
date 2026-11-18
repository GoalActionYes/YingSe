"""Test a closed-set style cue from reference luminance/chroma edge relationships."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_arri_sdr_renderability import PHOTOS, REPORT_SHA, digest, input_photo, metrics, quantize
from probe_crossed_edit_pair_supervision import BASE, ROLES, describe, fixed_sets, headroom, mask_for, pair, read
from reference_color_app.lut import _sample
from spatial_render import finish_spatial

LUMA = np.array([.2126, .7152, .0722])
ARMS = ("quantiles", "edges")
EDITS = ("off", "on")
CUBE_SHA = "6716a295fa14fc5cbbcef461dbd31c10c3df81babb2200eb9c561e05139e283f"


def features(rgb):
    raster = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float64)
    light = raster @ LUMA
    pairs = ((np.s_[:, :-4], np.s_[:, 4:]), (np.s_[:-4, :], np.s_[4:, :]))
    deltas, centers, chroma = [], [], []
    colors = raster[..., [0, 2]] - light[..., None]
    for first, second in pairs:
        deltas.append((light[second] - light[first]).ravel())
        centers.append(((light[first] + light[second]) / 2).ravel())
        chroma.append((colors[second] - colors[first]).reshape(-1, 2))
    dy, level, dc = map(np.concatenate, (deltas, centers, chroma))
    contrast = (np.abs(dy) >= 4) & (np.abs(dy) <= 64)
    slopes = []
    for lower in (0, 64, 128, 192):
        mask = contrast & (level >= lower) & (level < lower + 64)
        signal = dy[mask]
        energy = signal @ signal
        slopes.extend((signal @ dc[mask]) / energy if energy else (0., 0.))
    return describe(raster), np.asarray(slopes)


def render(rgb, cube):
    return quantize(_sample(cube, rgb.astype(np.float32) / 255))


def states(rgb, cube):
    return {"off": rgb, "on": render(rgb, cube)}


def train(fitting, cube):
    samples = [features(rgb) for group in fitting for image in group.values()
               for rgb in states(image, cube).values()]
    x = np.stack([np.concatenate(row) for row in samples])
    target = np.tile([0., 1.], len(samples) // 2)
    if len(samples) != 48:
        raise ValueError("Unexpected fit cohort size")
    count = len(samples[0][0])
    models = {}
    for arm, columns in zip(ARMS, (count, x.shape[1]), strict=True):
        train_x = x[:, :columns]
        center = train_x.mean(axis=0)
        spread = train_x.std(axis=0)
        scale = np.where(spread > 1e-12, spread, 1)
        z = (train_x - center) / scale
        weights = np.linalg.solve(z.T @ z + np.eye(columns), z.T @ (target - .5))
        models[arm] = (center, scale, weights)
    return models


def predict(reference, models):
    vector = np.concatenate(features(reference))
    return {arm: float((vector[:len(center)] - center) / scale @ weights + .5)
            for arm, (center, scale, weights) in models.items()}


def paint(target, cube_on, reference_states, models, folder):
    folder.mkdir()
    records, images = {}, {}
    for background, refs in reference_states.items():
        measurements = {edit: predict(ref, models) for edit, ref in refs.items()}
        for arm in models:
            records.setdefault(arm, {})[background] = {}
            branch = images.setdefault(arm, {}).setdefault(background, {})
            for edit in EDITS:
                score = measurements[edit][arm]
                file = folder / f"{arm}-{background}-{edit}.png"
                branch[edit] = cube_on if score >= .5 else target
                Image.fromarray(branch[edit]).save(file)
                records[arm][background][edit] = {"score": score, "is_film": score >= .5, "sha256": digest(file)}
    return records, images


def evaluate(target, expected, images, regions):
    return {background: {label: metrics(target, expected, row["off"], row["on"], mask)
                         for label, mask in regions.items() if mask.sum() >= 256}
            for background, row in images.items()}


def evaluate_group(target, expected, images, person, neutral):
    results = evaluate(target, expected, images, {"whole": np.ones(person.shape, bool), "neutral": neutral})
    gaps = {label: {edit: float(np.abs(images["C1"][edit][mask].astype(np.int16) - images["F1"][edit][mask]).mean())
                    for edit in EDITS} for label, mask in (("person", person), ("neutral", neutral)) if mask.sum() >= 256}
    passed = person.sum() >= 256 and all(row["passed"] for branch in results.values() for row in branch.values())
    passed = passed and all(gap <= 1.5 for region in gaps.values() for gap in region.values())
    return {"style": results, "swap_mae": gaps, "passed": bool(passed)}


def run(cube_file, real_folder, out):
    parent_file = real_folder / "results.json"
    if digest(cube_file) != CUBE_SHA or digest(parent_file) != REPORT_SHA:
        raise ValueError("Frozen LUT or real scenes changed")
    cube = np.loadtxt(cube_file, skiprows=4, dtype=np.float32).reshape(64, 64, 64, 3).transpose(2, 1, 0, 3)
    fitting, dev, targets = fixed_sets()
    models = train(fitting, cube)
    out.mkdir()
    groups = {}
    for row in dev.values():
        name = row["target_pair"]
        source = read(targets[name]["input_file"])
        target = headroom(source)
        expected = finish_spatial(target, render(target, cube), .8)
        person = mask_for(targets[name], name.split("-")[0])
        neutral = target_regions(source, person)["neutral_proxy"] & person
        refs = {key: states(value, cube) for key, value in pair(BASE / "bg-real-pixel-inputs", row["input"]).items()}
        logs, images = paint(target, expected, refs, models, out / name)
        groups[name] = {arm: {"choices": logs[arm], **evaluate_group(target, expected, images[arm], person, neutral)}
                        for arm in models}
    real = {}
    parent = json.loads(parent_file.read_bytes())
    native = {scene: {edit: input_photo(real_folder, f"{scene}-{edit}.png",
                                       parent["prepared"][scene]["images"][edit]) for edit in EDITS}
              for scene in PHOTOS}
    for scene, reference in (PHOTOS, PHOTOS[::-1]):
        original, film = (native[scene][key] for key in EDITS)
        expected = finish_spatial(original, film, .8)
        finished = finish_spatial(original, render(original, cube), .8)
        shape = original.shape[:2]
        neutral = np.logical_and(*(target_regions(x, np.zeros(shape, bool))["neutral_proxy"] for x in (original, film)))
        logs, images = paint(original, finished, {"native": native[reference]}, models, out / scene)
        digital_control = predict(render(native[reference]["off"], cube), models)
        real[scene] = {arm: {"choices": logs[arm]["native"], "style": evaluate(
            original, expected, images[arm], {"whole": np.ones(shape, bool), "neutral": neutral}
        )["native"], "same_reference_post_sdr_control_score": digital_control[arm]} for arm in models}

    counts = {arm: sum(row[arm]["passed"] for row in groups.values()) for arm in models}
    maximum_regression = max(
        row["edges"]["style"][background]["whole"][metric] - row["quantiles"]["style"][background]["whole"][metric]
        for row in groups.values() for background, metric in product(
            ROLES, ("off_matches_original_mae", "on_matches_authored_mae"))
    )
    maximum_regression = max(maximum_regression, max(
        row["edges"]["style"]["whole"][metric] - row["quantiles"]["style"]["whole"][metric]
        for row in real.values() for metric in ("off_matches_original_mae", "on_matches_authored_mae")
    ))
    real_counts = {arm: sum(all(region["passed"] for region in row[arm]["style"].values()) for row in real.values())
                   for arm in models}
    passed = counts["edges"] >= 3 and counts["edges"] > counts["quantiles"] and real_counts["edges"] == 2
    report = {"cube_sha256": CUBE_SHA, "real_source_sha256": REPORT_SHA, "fit_groups": len(fitting),
              "bg_real": groups, "native": real, "bg_passed": counts, "native_passed": real_counts,
              "worst_target_regression": maximum_regression, "passed": passed and maximum_regression <= 2}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdr_cube", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.sdr_cube, args.native, args.output)
