"""Compare per-image and paired edit objectives on frozen photograph groups."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, FIT_SHA, HELD_SHA, PARENT, PARENT_SHA, ROOT,
    describe, digest, feature, grade, headroom, mask_for, pair, panel, read, scores,
)
from probe_hdm_hdr_display import json_to

ROLES = ("C1", "F1")
Y = np.array([.2126, .7152, .0722])
ARMS = ("per-image", "paired")


def descriptor(rgb, local=True):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(image)
    global_values = describe(pixels)
    if not local:
        return global_values
    light = pixels @ Y
    padded = np.pad(light, ((0, 4), (0, 4)), mode="edge")
    h, w = light.shape
    diff = (np.abs(padded[4:, :w] - light) + np.abs(padded[:h, 4:] - light)) / 2
    colors = pixels[..., [0, 2]] - light[..., None]
    rank = np.argsort(light.ravel(), kind="stable")
    summary = []
    for group in np.array_split(rank, 4):
        summary.extend((*colors.reshape(-1, 2)[group].mean(axis=0), np.median(diff.ravel()[group])))
    return np.concatenate((global_values, summary))


def fixed_sets():
    folder = BASE / "bg-real-color-directions-inputs"
    train_file = folder / "intake.json"
    test_file = ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    parent_file = PARENT / "results.json"
    if (digest(train_file), digest(test_file), digest(parent_file)) != (FIT_SHA, HELD_SHA, PARENT_SHA):
        raise ValueError("Frozen development input changed")
    train = json.loads(train_file.read_bytes())
    dev = json.loads(test_file.read_bytes())["cases"]
    if len(train) != 12 or len(dev) != 4:
        raise ValueError("Unexpected number of source groups")
    if train.keys() & dev.keys() or {e["sha256"] for row in train.values() for e in row.values()} & {
        e["sha256"] for row in dev.values() for e in row["input"].values()
    }:
        raise ValueError("Training/reference overlap")
    saved = {row["pair"]: row for row in json.loads(parent_file.read_bytes())["cases"]}
    return [pair(folder, train[k]) for k in sorted(train)], dev, saved


def training_levels(fitting, local):
    targets = [descriptor(refs["C1"], local) for refs in fitting]
    levels = {}
    for group, refs in enumerate(fitting):
        assigned = targets[(group + 1) % len(fitting)]
        for role, name in product(ROLES, EDITS):
            levels[group, role, name] = feature(assigned, descriptor(grade(refs[role], EDITS[name]), local))
    return levels


def solve(fitting, local):
    levels = training_levels(fitting, local)
    x = np.stack(list(levels.values()))
    y = np.asarray([EDITS[name] for _, _, name in levels], dtype=np.float64)
    center, spread, intercept = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
    scale = np.where(spread > 0, spread, 1.)
    level = ((x - center) / scale, y - intercept)
    toggles = [(levels[group, role, name] - levels[group, role, "none"], EDITS[name])
               for group, role, name in product(range(len(fitting)), ROLES, EDITS) if name != "none"]
    swaps = [levels[group, "F1", name] - levels[group, "C1", name]
             for group, name in product(range(len(fitting)), EDITS)]
    toggle = (np.stack([row[0] for row in toggles]) / scale, np.stack([row[1] for row in toggles]))
    swap = (np.stack(swaps) / scale, np.zeros((len(swaps), y.shape[1])))
    penalty = np.eye(x.shape[1]) / len(x)
    terms = (level, toggle, swap)
    outputs, audit = {}, {}
    for arm, weight in zip(ARMS, ((1., 0., 0.), (.5, .25, .25)), strict=True):
        normal = penalty + sum(w * a.T @ a / len(a) for w, (a, _) in zip(weight, terms, strict=True))
        rhs = sum(w * a.T @ b / len(a) for w, (a, b) in zip(weight, terms, strict=True))
        weights = np.linalg.solve(normal, rhs)
        outputs[arm] = (center, scale, intercept, weights)
        audit[arm] = {key: float(np.abs(a @ weights - b).mean())
                      for key, (a, b) in zip(("level_mae", "toggle_mae", "swap_mae"), terms, strict=True)}
        audit[arm]["normal_equation_residual"] = float(np.max(np.abs(normal @ weights - rhs)))
    return outputs, {"level_rows": len(x), "toggle_rows": len(toggles), "swap_rows": len(swaps),
                     "features": x.shape[1], "fit": audit}


def estimate(model, target_features, reference_features):
    center, scale, intercept, weights = model
    features = feature(target_features, reference_features)
    return np.clip(((features - center) / scale) @ weights + intercept, (-1., 0.), (1., 1.))


def run(out, local=True, fit_models=None, candidate="paired"):
    train, dev, registered = fixed_sets()
    if fit_models is None:
        models, audit = solve(train, local)
        feature_audit = None
    else:
        models, audit, feature_audit = fit_models(train, local)
    out.mkdir()
    cases = {}
    deepest = -np.inf
    for row in dev.values():
        key = row["target_pair"]
        orig = read(registered[key]["input_file"])
        target = headroom(orig)
        df = descriptor(target, local)
        person = mask_for(registered[key], key.split("-")[0])
        neutral = target_regions(orig, person)["neutral_proxy"] & person
        refs = pair(BASE / "bg-real-pixel-inputs", row["input"])
        truth = {name: grade(target, coeff, .8) for name, coeff in EDITS.items()}
        observed = {(role, name): descriptor(grade(refs[role], coeff), local)
                    for role, (name, coeff) in product(ROLES, EDITS.items())}
        outputs, values = {"Prescribed edit": truth}, {}
        folder = out / key
        folder.mkdir()
        for arm, model in models.items():
            photos = {}
            picks, hashes = {}, {}
            for role in ROLES:
                photos[role], picks[role] = {}, {}
                for name in EDITS:
                    coeff = estimate(model, df, observed[role, name])
                    rendered = grade(target, coeff, .8)
                    file = folder / f"{arm}-{role}-{name}.png"
                    Image.fromarray(rendered).save(file)
                    hashes[file.name] = digest(file)
                    picks[role][name] = coeff.tolist()
                    photos[role][name] = rendered
                outputs[f"{arm} / {role}"] = photos[role]
            values[arm] = {**scores(photos, truth, person, neutral), "coefficients": picks, "hashes": hashes}
        for role, name in product(ROLES, EDITS):
            left, right = (values[arm]["styles"][name]["response"][role]["target_mae"]
                           for arm in ("per-image", candidate))
            deepest = max(deepest, right - left)
        picture = folder / "comparison.png"
        panel(picture, outputs)
        cases[key] = {**values, "comparison_sha256": digest(picture)}
        if feature_audit is not None:
            cases[key]["feature_audit"] = feature_audit(observed)
    count = {arm: sum(row[arm]["passed"] for row in cases.values()) for arm in models}
    json_to(out / "results.json", {"inputs": {"fit": FIT_SHA, "held": HELD_SHA, "analysis_parent": PARENT_SHA},
                                   "training": audit, "cases": cases, "passed_by_arm": count,
                                   "worst_target_regression": deepest,
                                   "passed": count[candidate] >= 3 and count[candidate] > count["per-image"]
                                   and deepest <= 2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quantiles-only", action="store_true")
    args = parser.parse_args()
    run(args.output, not args.quantiles_only)
