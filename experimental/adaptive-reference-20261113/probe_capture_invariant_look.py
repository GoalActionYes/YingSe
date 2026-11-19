"""Test a paired physical-variation penalty on known looks across held-out sources."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
from probe_arri_two_looks import ASSIGNMENTS
from probe_dast_local_edit_cross import INPUT_SHA
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_middlebury_camera_calibration_order import load_reference
from probe_middlebury_camera_channel_replay import CROSS_SHA
from probe_middlebury_camera_film_cross import PARENT_SHA
from probe_mls_film_light_cross import SOURCE_SHA
from probe_psbench_background_desaturation import REC709
from spatial_render import finish_spatial


LABELS = ("off", "film_a", "film_b")
ARMS = {"ridge": 0., "paired": 1.}
FILM_B_SHA = "ef7a8ed92baf4109236131bbb51b3a55a9bbeaff980577f2df845d418096a30a"


def style_transforms(base):
    table = {"film_a": (base / "arri-shared-look-assets-v1/film-a.cube", LOOK_SHA),
             "film_b": (base / "arri-second-film-assets-v1/second-film.cube", FILM_B_SHA)}
    result = {}
    for label, (path, expected) in table.items():
        if digest(path) != expected:
            raise ValueError(f"Changed {label} transform")
        result[label] = transforms(path)
    return result


def render_style(rgb, transform):
    view, inverse, grade = transform
    log = converted(rgb.astype(np.float32) / 255, inverse)
    return quantize(converted(converted(log, grade), view))


def natural_scenes(base):
    root = base / "arri-two-looks-v2"
    ledger = verified_json(root / "results.json", PARENT_SHA)
    scenes = {}
    for scene in ASSIGNMENTS:
        names = {"off": ("film_a", "off"), "film_a": ("film_a", "on"), "film_b": ("film_b", "on")}
        scenes[scene] = {name: verified(root / family / f"{scene}-{role}.png",
                                        ledger["preparation"][family][scene]["images"][role])
                         for name, (family, role) in names.items()}
    return scenes


def capture_groups(base):
    dast = base / "dast-exposure-prepared-v1"
    mls = base / "mls-linear-pilot"
    dast_report = verified_json(dast / "inputs.json", INPUT_SHA)
    mls_report = verified_json(mls / "results.json", SOURCE_SHA)
    cross = verified_json(base / "middlebury-camera-film-cross-v1/results.json", CROSS_SHA)
    return {
        "dast": {str(row["group"]): {state: verified(dast / str(row["group"]) / f"{state}.png", row["frames"][state])
                                    for state in ("nominal", "ue1", "oe1")} for row in dast_report["groups"]},
        "mls": {scene: {state: verified(mls / f"{scene}_2HAL_DESK{suffix}.png",
                                     mls_report["scenes"][scene]["inputs_sha256"][f"{scene}_2HAL_DESK{suffix}.png"])
                       for state, suffix in (("standard", ""), ("blue", "_LED-B025"))} for scene in ("02", "03")},
        "middlebury": {"chart": {key: value["off"] for key, value in load_reference(base, cross).items()}},
    }


def descriptor(rgb):
    light = rgb @ REC709
    values = np.stack((light, rgb[..., 0] - light, rgb[..., 2] - light), axis=-1).reshape(-1, 3)
    result = np.concatenate((np.quantile(values, np.arange(1, 10) / 10, axis=0),
                             values.mean(0)[None], values.std(0)[None]), axis=0)
    return result.ravel()


def samples(base, scenes, transforms):
    cohorts = capture_groups(base)
    data = {}
    for domain, groups in cohorts.items():
        data[domain] = {}
        for group, views in groups.items():
            data[domain][group] = {}
            for name, rgb in views.items():
                guide = Image.fromarray(rgb)
                guide.thumbnail((512, 512), Image.Resampling.BILINEAR)
                original = np.array(guide)
                data[domain][group][name] = {"off": descriptor(original)} | {
                    label: descriptor(render_style(original, transform)) for label, transform in transforms.items()
                }
    data["arri"] = {scene: {"native": {label: descriptor(rgb) for label, rgb in states.items()}}
                    for scene, states in scenes.items()}
    return data


def fit(folds, penalty):
    rows, labels, nuisance = [], [], []
    for cohort in folds:
        for views in cohort.values():
            for sample in views.values():
                for label in LABELS:
                    rows.append(sample[label])
                    labels.append(LABELS.index(label))
            for left, right in combinations(views.values(), 2):
                nuisance.extend(left[label] - right[label] for label in LABELS)
    raw = np.array(rows)
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 1e-6)
    designs = (raw - center) / scale
    paired_delta = np.asarray(nuisance) / scale
    truth = np.eye(len(LABELS))[labels]
    prior = truth.mean(axis=0)
    regularizer = np.eye(designs.shape[1]) + penalty * (paired_delta.T @ paired_delta) / len(paired_delta)
    weights = np.linalg.solve(designs.T @ designs / len(designs) + regularizer,
                              designs.T @ (truth - prior) / len(designs))
    return center, scale, prior, weights, len(raw), len(nuisance)


def guess(feature, model):
    center, scale, prior, weights, _, _ = model
    values = (feature - center) / scale @ weights + prior
    index = int(np.argmax(values))
    chosen = LABELS[index] if values[index] >= .65 else "off"
    return chosen, values.tolist()


def target_scene(domain, group):
    if domain == "arri":
        return ASSIGNMENTS[group]
    return "hdr_testimage" if domain == "mls" else "poker_fullshot"


def renderings(scenes, transforms):
    results = {}
    for scene, inputs in scenes.items():
        target = inputs["off"]
        actual = {"off": target} | {label: finish_spatial(target, render_style(target, transform), .8)
                                     for label, transform in transforms.items()}
        expected = {label: finish_spatial(target, rgb, .8) for label, rgb in inputs.items()}
        whole = np.ones(target.shape[:2], bool)
        neutral = np.logical_and.reduce([target_regions(rgb, ~whole)["neutral_proxy"] for rgb in inputs.values()])
        if neutral.sum() < 256:
            raise ValueError("Lost whole/neutral support")
        results[scene] = actual, expected, {"whole": whole, "neutral": neutral}
    return results


def evaluate(cohort, domain, model, prepared, out):
    observations, stability = {}, {}
    all_hits = identity_count = passed_views = count_views = 0
    for group, views in cohort.items():
        actual, truth, masks = prepared[target_scene(domain, group)]
        choices = {}
        for view, samples in views.items():
            count_views += 1
            stems, estimates = {}, {}
            for label, data in samples.items():
                chosen, values = guess(data, model)
                result = actual[chosen]
                path = out / f"{group}-{view}-{label}.png"
                Image.fromarray(result).save(path)
                estimates[label], stems[label] = chosen, {"choice": chosen, "scores": values, "sha256": digest(path)}
                all_hits += chosen == label
                if label == "off":
                    identity_count += np.array_equal(result, actual["off"])
            choices[view] = estimates
            style = {}
            for left, right in zip(LABELS, LABELS[1:]):
                style[f"{left}_to_{right}"] = {
                    key: metrics(truth[left], truth[right], actual[estimates[left]], actual[estimates[right]], area)
                    for key, area in masks.items()
                }
            passed_views += all(row["passed"] for change in style.values() for row in change.values())
            observations[f"{group}-{view}"] = {"decisions": stems, "styles": style}
        neutral = masks["neutral"]
        for (a, lhs), (b, rhs) in combinations(choices.items(), 2):
            stability[f"{group}-{a}-{b}"] = {
                label: float(np.abs(actual[lhs[label]][neutral].astype(float) - actual[rhs[label]][neutral]).mean())
                for label in LABELS
            }
    worst = max((value for row in stability.values() for value in row.values()), default=0)
    passed = (all_hits >= .8 * count_views * len(LABELS) and identity_count == count_views
              and passed_views >= .8 * count_views and worst <= 3)
    return {"correct": all_hits, "states": count_views * len(LABELS), "off_identity": identity_count,
            "style_views_passed": passed_views, "views": count_views, "worst_same_style_neutral_mae": worst,
            "observations": observations, "same_style": stability, "passed": passed}


def run(base, out):
    transforms = style_transforms(base)
    scenes = natural_scenes(base)
    data = samples(base, scenes, transforms)
    outputs = renderings(scenes, transforms)
    count = sum(len(views) for domain in data.values() for views in domain.values())
    if count != 24:
        raise ValueError("Changed frozen cohort")
    out.mkdir()
    report = {}
    for name, penalty in ARMS.items():
        report[name] = {}
        for domain, cohort in data.items():
            folder = out / name / domain
            folder.mkdir(parents=True)
            model = fit([others for key, others in data.items() if key != domain], penalty)
            report[name][domain] = evaluate(cohort, domain, model, outputs, folder) | {
                "train_states": model[-2], "train_differences": model[-1],
            }
    passes = {arm: sum(entry["passed"] for entry in fold.values()) for arm, fold in report.items()}
    json_to(out / "results.json", {"sources": count, "arms": report, "passing_domains": passes,
                                   "passed": passes["paired"] >= 3 and passes["paired"] > passes["ridge"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
