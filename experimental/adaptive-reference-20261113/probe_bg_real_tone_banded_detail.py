"""Add tone-banded relative local contrast to the existing two-image edit fit."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from probe_crossed_edit_pair_supervision import (
    BASE, EDITS, FIT_SHA, HELD_SHA, ROLES, Y, descriptor, digest, estimate,
    feature, fixed_sets, grade, headroom, json_to, mask_for, pair, panel, read, scores, target_regions,
)
from probe_bg_real_crossed_grade_training import fit
from probe_physical_blue_global_fit import linear


ARMS = ("control", "banded")
BANDS = ((32, 96), (96, 160), (160, 224))
QUANTILES = (50, 75)


def local_detail(rgb):
    picture = Image.fromarray(rgb)
    picture.thumbnail((256, 256), Image.Resampling.BILINEAR)
    sample = np.asarray(picture)
    display_y = sample @ Y
    log_y = np.log(np.maximum(linear(sample) @ Y, .001))
    pairs = ((np.s_[:, 4:], np.s_[:, :-4]), (np.s_[4:, :], np.s_[:-4, :]))
    tone = np.concatenate([((display_y[a] + display_y[b]) * .5).ravel() for a, b in pairs])
    delta = np.concatenate([np.abs(log_y[a] - log_y[b]).ravel() for a, b in pairs])
    floor = np.maximum(np.percentile(delta, QUANTILES), .001) if delta.size else np.ones(2)
    result, count = [], []
    for lo, hi in BANDS:
        selected = delta[(tone >= lo) & (tone < hi)]
        count.append(len(selected))
        result.extend((np.maximum(np.percentile(selected, QUANTILES), .001) / floor)
                      if len(selected) >= 64 else (1., 1.))
    return np.array(result), count


def describe_both(rgb):
    standard = descriptor(rgb)
    detail, count = local_detail(rgb)
    return {"control": standard, "banded": np.concatenate((standard, detail))}, count


def models_for(groups):
    targets = [describe_both(group["C1"])[0] for group in groups]
    samples = {arm: [] for arm in ARMS}
    truths = []
    for i, group in enumerate(groups):
        target = targets[(i + 1) % len(targets)]
        for role, operation in product(ROLES, EDITS.values()):
            reference, _ = describe_both(grade(group[role], operation))
            for arm in ARMS:
                samples[arm].append(feature(target[arm], reference[arm]))
            truths.append(operation)
    if len(truths) != 96:
        raise ValueError("Unexpected fitting budget")
    return {arm: fit(values, truths) for arm, values in samples.items()}


def render(model, arm, target, reference, target_features=None, reference_features=None):
    if np.array_equal(target, reference):
        return target.copy(), [0., 0.]
    if target_features is None:
        target_features = describe_both(target)[0][arm]
    if reference_features is None:
        reference_features = describe_both(reference)[0][arm]
    parameters = estimate(model, target_features, reference_features)
    return grade(target, parameters, .8), parameters.tolist()


def run(out):
    fitting, development, registered = fixed_sets()
    models = models_for(fitting)
    flat = np.full((32, 32, 3), 128, np.uint8)
    detail, count = local_detail(flat)
    np.testing.assert_array_equal(detail, np.ones(6))
    if count[0] != 0 or count[1] < 64 or count[2] != 0:
        raise ValueError("Flat-image fallback covered the wrong tone band")
    for arm, model in models.items():
        np.testing.assert_array_equal(render(model, arm, flat, flat.copy())[0], flat)
    cases = {}
    out.mkdir()
    for row in development.values():
        key = row["target_pair"]
        entry = registered[key]
        source = read(entry["input_file"])
        target = headroom(source)
        target_features, target_support = describe_both(target)
        refs = pair(BASE / "bg-real-pixel-inputs", row["input"])
        folder = out / key
        folder.mkdir()
        outputs, artifacts = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
        support = {}
        for role, original in refs.items():
            for arm in ARMS:
                outputs[arm][role], artifacts[arm][role] = {}, {}
            for name, operation in EDITS.items():
                reference = grade(original, operation)
                feature_values, support[f"{role}-{name}"] = describe_both(reference)
                for arm in ARMS:
                    frame, parameters = render(models[arm], arm, target, reference,
                                               target_features[arm], feature_values[arm])
                    outputs[arm][role][name] = frame
                    path = folder / f"{arm}-{role}-{name}.png"
                    Image.fromarray(frame).save(path)
                    artifacts[arm][role][name] = {"parameters": parameters, "sha256": digest(path)}
        truth = {name: grade(target, operation, .8) for name, operation in EDITS.items()}
        person = mask_for(entry, key.split("-")[0])
        neutral = target_regions(source, person)["neutral_proxy"] & person
        evaluated = {arm: scores(pics, truth, person, neutral) for arm, pics in outputs.items()}
        excess = max(evaluated["banded"]["styles"][name]["response"][role]["target_mae"]
                     - evaluated["control"]["styles"][name]["response"][role]["target_mae"]
                     for name, role in product(EDITS, ROLES))
        board = folder / "comparison.png"
        panel(board, {"Prescribed edit": truth, **{f"{arm} / {role}": pics for arm, rows in outputs.items()
                                                   for role, pics in rows.items()}})
        cases[key] = {"scores": evaluated, "max_extra_mae": excess, "hashes": artifacts,
                      "target_tone_support": target_support, "reference_tone_support": support,
                      "comparison_sha256": digest(board)}
    count = {arm: sum(row["scores"][arm]["passed"] for row in cases.values()) for arm in ARMS}
    worst = max(row["max_extra_mae"] for row in cases.values())
    json_to(out / "results.json", {"fit_sha256": FIT_SHA, "development_sha256": HELD_SHA,
                                  "counts": count, "worst_target_regression": worst, "cases": cases,
                                  "passed": count["banded"] >= 3 and count["banded"] > count["control"] and worst <= 2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
