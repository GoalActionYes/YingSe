"""Fit two-image render responses with and without matched capture supervision."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_capture_invariant_look import (
    LABELS, capture_groups, fit, guess, natural_scenes, render_style, style_transforms, target_scene,
)
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_gros_render_intent_projection import grade
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

ARMS = {"direct_fit": 0., "paired_fit": 1.}
STYLES = LABELS[1:]


def parameters(engine, target, ops):
    self_grade = grade(engine, target, target)
    frames = {"off": target} | {
        style: finish_spatial(target, render_style(target, ops[style]), STRENGTH) for style in STYLES
    }
    templates = np.stack([frames[style].astype(np.float64) - target for style in STYLES])
    luma = target @ np.array([.2126, .7152, .0722])
    strata = np.digitize(luma, np.quantile(luma, [.25, .5, .75]))
    return self_grade, frames, templates, strata


def describe(target_grade, self_grade, templates, strata):
    delta = target_grade.astype(np.float64) - self_grade
    feature = []
    for band in range(4):
        area = strata == band
        basis = templates[:, area]
        numerator = np.sum(basis * delta[area], axis=1)
        denominator = np.maximum(np.sum(basis**2, axis=1), 1)
        feature.extend((numerator / denominator).ravel())
    return np.asarray(feature)


def samples(engine, ops):
    scenes = natural_scenes(BASE)
    training = {}
    for domain, groups in capture_groups(BASE).items():
        target = scenes[target_scene(domain, next(iter(groups)))]["off"]
        self_grade, _, templates, strata = parameters(engine, target, ops)
        training[domain] = {}
        for name, views in groups.items():
            training[domain][name] = {}
            for view, raw in views.items():
                image = Image.fromarray(raw)
                image.thumbnail((384, 384), Image.Resampling.BILINEAR)
                reference = np.array(image)
                inputs = {"off": reference} | {style: render_style(reference, ops[style]) for style in STYLES}
                training[domain][name][view] = {
                    label: describe(grade(engine, target, frame), self_grade, templates, strata)
                    for label, frame in inputs.items()
                }
    return training


def choose(target, reference, feature, model, frames):
    if np.array_equal(target, reference):
        return target.copy(), "off", [1., 0., 0.]
    label, raw = guess(feature, model)
    return frames[label], label, raw


def run(out):
    source = BASE / "gros-film-b-cross-v2"
    archive = checked_json(source / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    ops = style_transforms(BASE)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or archive["strength"] != STRENGTH:
        raise ValueError("Changed frozen 13 output contract")
    out.mkdir()
    training = samples(engine, ops)
    models = {name: fit(training.values(), penalty) for name, penalty in ARMS.items()}
    photos = {}
    for key, identity in archive["inputs"].items():
        original, film_b = [
            checked_rgb(source / f"{key}-{state}.png", identity["hashes"][state]) for state in ("off", "on")
        ]
        photos[key] = {"off": original, "film_a": render_style(original, ops["film_a"]), "film_b": film_b}
    contexts = {person: parameters(engine, photos[f"{person}-cold"]["off"], ops) for person in GROUPS}
    cases = {}
    for person, donor in permutations(GROUPS, 2):
        self_grade, expected, templates, strata = contexts[person]
        target = photos[f"{person}-cold"]["off"]
        areas = masks(labels[person]["cold"], labels["screen"], target.shape[:2])
        name = f"{person}-from-{donor}"
        frames = {arm: {light: {} for light in LIGHTS} for arm in ("frozen13", *ARMS)}
        decisions, hashes = {}, {}
        for arm, model in models.items():
            same, choice, _ = choose(target, target, np.zeros(24), model, expected)
            if choice != "off" or not np.array_equal(same, target):
                raise ValueError("Same-image preservation failed")
        for light in LIGHTS:
            for role, reference in photos[f"{donor}-{light}"].items():
                rendered = grade(engine, target, reference)
                frames["frozen13"][light][role] = rendered
                feature = describe(rendered, self_grade, templates, strata)
                for arm, model in models.items():
                    image, choice, confidence = choose(target, reference, feature, model, expected)
                    frames[arm][light][role] = image
                    key = f"{name}-{arm}-{light}-{role}"
                    hashes[key] = save(out / f"{key}.png", image)
                    decisions[key] = dict(label=choice, scores=confidence)
        scores = {arm: {style: score(target, expected[style], {
            light: {"off": variants["off"], "on": variants[style]} for light, variants in lights.items()
        }, areas) for style in STYLES} for arm, lights in frames.items()}
        board = panel(out / f"{name}-comparison.png", {"expected": expected} | {
            f"{arm} {light}": variants for arm, lights in frames.items() for light, variants in lights.items()
        })
        cases[name] = dict(scores=scores, decisions=decisions, hashes=hashes, board=board)
    protocol = Path(__file__).with_name("gros-response-supervision-protocol.md")
    source_sizes = {name: sum(len(views) * len(LABELS) for views in groups.values()) for name, groups in training.items()}
    json_to(out / "results.json", dict(parent=BASELINE_SHA, regions=REGIONS_SHA, checkpoint=CHECKPOINT,
              strength=STRENGTH, protocol=digest(protocol), training=source_sizes,
              comparisons={arm: int(model[-1]) for arm, model in models.items()}, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
