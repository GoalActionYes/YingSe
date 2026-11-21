"""Screen an inverse-filter residual as a closed-set editing feature."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from probe_capture_invariant_look import (
    LABELS, descriptor, fit, guess, natural_scenes, render_style, samples, style_transforms,
)
from probe_gros_cifr_edit_replay import ASSETS, load, restore, small
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, finish_spatial, json_to, save

ARMS = ("observed", "residual")


def describe(image, encoder, vgg):
    reduced = small(image)
    estimated, _ = restore(encoder, vgg, reduced)
    delta = reduced.astype(np.int16) - estimated
    return dict(observed=descriptor(reduced), residual=descriptor(delta))


def fit_arm(training, arm):
    features = [
        {group: {variant: {label: by_arm[arm] for label, by_arm in frame.items()}
                 for variant, frame in views.items()} for group, views in domain.items()}
        for domain in training.values()
    ]
    return fit(features, 1.)


def run(out):
    photos = BASE / "gros-film-b-cross-v2"
    archive = checked_json(photos / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    encoder, vgg, tensor_counts = load(BASE / "cifr-intake-v1")
    transforms = style_transforms(BASE)

    def describe_image(image):
        return describe(image, encoder, vgg)

    train = samples(BASE, natural_scenes(BASE), transforms, describe=describe_image)
    models = {arm: fit_arm(train, arm) for arm in ARMS}
    if any(model[3].shape != (33, 3) or model[4] != 72 for model in models.values()):
        raise ValueError("Unexpected training dimensions or case count")

    originals, decisions = {}, {}
    feature_count = 0
    for person in GROUPS:
        decisions[person] = {}
        for light in LIGHTS:
            key = f"{person}-{light}"
            states = archive["inputs"][key]["hashes"]
            original = checked_rgb(photos / f"{key}-off.png", states["off"])
            edited = checked_rgb(photos / f"{key}-on.png", states["on"])
            if light == "cold":
                originals[person] = original
            refs = dict(off=original, film_a=render_style(original, transforms["film_a"]), film_b=edited)
            feature_count += len(refs)
            features = {label: describe_image(image) for label, image in refs.items()}
            decisions[person][light] = {
                label: {arm: guess(data[arm], models[arm]) for arm in ARMS}
                for label, data in features.items()
            }
    if feature_count != 18:
        raise ValueError("Unexpected Gros test count")

    out.mkdir()
    grades = {}
    for person, original in originals.items():
        grades[person] = {"off": original} | {
            label: finish_spatial(original, render_style(original, operation), STRENGTH)
            for label, operation in transforms.items()
        }
    cases = {}
    for person, donor in permutations(GROUPS, 2):
        name = f"{person}-from-{donor}"
        original = originals[person]
        area = masks(labels[person]["cold"], labels["screen"], original.shape[:2])
        expected = checked_rgb(photos / f"{name}-expected.png", archive["cases"][name]["expected_sha"])
        if not np.array_equal(expected, grades[person]["film_b"]):
            raise ValueError("Changed Film B positive control")
        cases[name] = {}
        for arm in ARMS:
            frames, files = {}, {}
            for light in LIGHTS:
                choice = decisions[donor][light]
                frames[light] = {stage: grades[person][choice[label][arm][0]]
                                 for stage, label in (("off", "off"), ("on", "film_b"))}
                for stage, pixels in frames[light].items():
                    suffix = f"{light}-{stage}"
                    files[suffix] = save(out / f"{name}-{arm}-{suffix}.png", pixels)
            cases[name][arm] = dict(scores=score(original, expected, frames, area), hashes=files)
    results = {
        arm: {label: sum(values[label][arm][0] == label for person in decisions.values()
                         for values in person.values()) for label in LABELS}
        for arm in ARMS
    }
    json_to(out / "results.json", dict(
        inputs=BASELINE_SHA, assets=ASSETS, tensor_counts=tensor_counts, training_examples=models["observed"][4],
        capture_pairs=models["observed"][5], features=33, total_votes=results, decisions=decisions, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
