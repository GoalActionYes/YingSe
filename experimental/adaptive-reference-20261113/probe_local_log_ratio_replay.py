"""Generate continuous two-image color edits from local log-channel ratios."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np

from probe_capture_invariant_look import capture_groups, natural_scenes, render_style, samples, style_transforms
from probe_capture_ordinal_edges import sets
from probe_gros_channel_envelope import BASELINE_SHA, REGRESSION_SHA, STATES
from probe_gros_film_b_cross import LIGHTS, score
from probe_gros_two_image_edit_regression import coefficients, feature_pair, model, predict, small
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save

QUANTILES = (.10, .25, .50, .75, .90)


def differences(log_rgb, valid, step):
    selected = []
    for axis in (0, 1):
        color = np.moveaxis(log_rgb, axis, 0)
        available = np.moveaxis(valid, axis, 0)
        delta = color[step:] - color[:-step]
        green = delta[..., 1]
        support = available[step:] & available[:-step] & (np.abs(green) >= .08)
        direction = np.sign(green[support])[:, None]
        selected.append((delta[support][:, [0, 2]] - green[support, None]) * direction)
    return np.concatenate(selected)


def describe(image):
    rgb = small(image)
    values = rgb.astype(float) / 255
    linear = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055)**2.4)
    logs = np.log(np.maximum(1e-6, linear))
    valid = (rgb.min(axis=-1) >= 12) & (rgb.max(axis=-1) <= 243)
    blocks = [differences(logs, valid, step) for step in (2, 8)]
    return np.concatenate([np.quantile(col, QUANTILES, axis=0).ravel() if len(col) >= 128 else np.zeros(10)
                           for col in blocks])


def fit_ratios(ops):
    natural = natural_scenes(BASE)
    references = samples(BASE, natural, ops, describe=describe)
    targets = {}
    for domain, cohort in capture_groups(BASE).items():
        for name, views in cohort.items():
            target = small(next(iter(views.values())))
            rendered = {key: render_style(target, op) for key, op in ops.items()}
            targets[domain, name] = target, rendered
    for name, views in natural.items():
        targets["arri", name] = views["off"], {key: views[key] for key in ops}
    desired = {key: (describe(rgb), {"off": np.zeros((10, 3))} |
                      {role: coefficients(rgb, frame) for role, frame in states.items()})
               for key, (rgb, states) in targets.items()}
    rows, labels = [], []
    for domain, cohort in references.items():
        for name, views in cohort.items():
            for view in views.values():
                for key, (target, desired_coefficients) in desired.items():
                    if key == (domain, name):
                        continue
                    for state, reference in view.items():
                        rows.append(feature_pair(target, reference))
                        labels.append(desired_coefficients[state].ravel())
    return model(rows, labels), len(rows)


def controls():
    x = np.arange(256, dtype=float).reshape(16, 16)
    logs = np.stack((.025*x, .035*x, .028*x), axis=-1) - 9.
    valid = np.ones(x.shape, bool)
    gain = np.log([1.24, .81, 1.4])
    for step in (2, 8):
        original = differences(logs, valid, step)
        shifted = differences(logs + gain, valid, step)
        if len(original) < 128:
            raise ValueError("Gain test lacks valid pairs")
        np.testing.assert_allclose(original, shifted, atol=1e-14, rtol=0)


def run(out):
    controls()
    ops = style_transforms(BASE)
    params, rows = fit_ratios(ops)
    archive = BASE / "gros-two-image-edit-regression-v1"
    old = checked_json(archive / "results.json", REGRESSION_SHA)
    if old["parent"] != BASELINE_SHA or old["training_pairs"] != rows:
        raise ValueError("Continuous reference training setup changed")
    out.mkdir()
    suites = {}
    for domain, inputs, regions, _ in sets(ops):
        folder = out / domain
        folder.mkdir()
        descriptors = {name: {light: {role: describe(image) for role, image in views.items()}
                              for light, views in lights.items()} for name, lights in inputs.items()}
        cases = {}
        for person, donor in permutations(inputs, 2):
            key = f"{person}-from-{donor}"
            source = inputs[person]["cold"]["off"]
            target_feature = descriptors[person]["cold"]["off"]
            expected = finish_spatial(source, inputs[person]["cold"]["film_b"], STRENGTH)
            identity = predict(source, source, target_feature, target_feature, params, True)
            np.testing.assert_array_equal(identity, source)
            frames, artifacts = {}, {}
            for light in LIGHTS:
                frames[light] = {}
                for state, role in zip(STATES, ("off", "film_b"), strict=True):
                    reference = inputs[donor][light][role]
                    feature = descriptors[donor][light][role]
                    image = predict(source, reference, target_feature, feature, params, True)
                    frames[light][state] = image
                    artifacts[f"{light}-{state}"] = save(folder / f"{key}-{light}-{state}.png", image)
            scores = score(source, expected, frames, regions[person])
            if domain == "portrait":
                prior = old["cases"][key]
                baselines = {light: {state: checked_rgb(archive / f"{key}-two_image-{light}-{role}.png",
                            prior["hashes"][f"two_image-{light}-{role}"])
                            for state, role in zip(STATES, ("off", "film_b"), strict=True)} for light in LIGHTS}
                base_scores = score(source, expected, baselines, regions[person])
                if base_scores != prior["scores"]["two_image"]["film_b"]:
                    raise ValueError("Comparison scores changed")
            else:
                archive_object = BASE / "openillumination-fixed-cross-v1"
                ledger = checked_json(archive_object / "results.json", "6b9f2c7af57709c4f04b236a3a798217777f17a85724121fd9bb32beefdf0d02")
                prior = ledger["cases"][key]
                baselines = {light: {state: checked_rgb(archive_object / f"{key}-{tag}-{state}.png",
                            prior["hashes"][f"{tag}-{state}"]) for state in STATES}
                            for light, tag in zip(LIGHTS, ("001", "013"), strict=True)}
                base_scores = score(source, expected, baselines, regions[person])
                if base_scores != prior["scores"]:
                    raise ValueError("Comparison scores changed")
            board = {"target": {"original": source, "expected": expected}}
            for light in LIGHTS:
                board[f"control {light}"] = baselines[light]
                board[f"ratios {light}"] = frames[light]
            cases[key] = dict(scores=scores, baseline=base_scores, artifacts=artifacts,
                              board=panel(folder / f"{key}-comparison.png", board))
        count = sum(all(r["passed"] for r in case["scores"].values()) for case in cases.values())
        lost = [(name, zone) for name, case in cases.items() for zone, value in case["baseline"].items()
                if value["passed"] and not case["scores"][zone]["passed"]]
        suites[domain] = dict(cases=cases, passes=count, lost=lost, qualified=count >= 5 and not lost)
    protocol = Path(__file__).with_name("local-log-ratio-replay-protocol.md")
    json_to(out / "results.json", dict(portrait=BASELINE_SHA, continuous=REGRESSION_SHA,
                                       training_pairs=rows, strength=STRENGTH, protocol=digest(protocol), suites=suites))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
