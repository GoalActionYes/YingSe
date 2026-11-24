"""Read out daylight/edit distinctions at frozen reference and grid representations."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from probe_cross_source_wall_lightening_workbench import DIRECTIONS, frozen_scenes, load
from probe_target_mirror_chroma import CHECKPOINT, digest, json_to
from reference_patch import reference_tokens
from spatial_render import SpatialEngine
from train_positional_lightness import ConditionedEngine, FAMILIES


PROTOCOL = "107ae4d15e87921dc4add5f288b1fc16a5427d887fbf0574e48b415b26d7baee"
TRAINING = "32a034d2c6bd7f52a50699368a795c0c9bc44890e1a93ad5a20464dabdbb4893"
CONDITIONS = ("rgb4", "histogram", "projected4", "grid13", "grid_local")
REGULARIZATION = .1


def read_frames(scenes):
    pictures, target = {}, {}
    for left, right in FAMILIES.values():
        for donor, subject in ((left, right), (right, left)):
            for direction in DIRECTIONS[donor]:
                a, b = direction.split("-to-")
                for tag in (f"{a}-off", f"{b}-off", f"{a}-on"):
                    if (donor, tag) not in pictures:
                        pictures[donor, tag] = load(scenes[donor], tag)[1]
            target[donor] = load(scenes[subject], f"{scenes[subject]['anchor']}-off")[1]
    return pictures, target


@torch.inference_mode()
def features(engine, old, pictures, target):
    output = {mode: [] for mode in CONDITIONS}
    for donor, tag in pictures:
        analysis = old.analysis(pictures[donor, tag])
        context = old.analysis(target[donor])
        projected = old.model.img_projection(old.api["TF"].resize(analysis, (64, 64), antialias=True))
        bins = old.api["batch_histogram_1D"](old.api["srgb_to_ycbcr"](analysis))
        target_bins = old.api["batch_histogram_1D"](old.api["srgb_to_ycbcr"](context))
        main = old.api["TF"].resize(context, (64, 64), antialias=True)
        grids = old.api["lowpass"](old.model.apply_grid(main, torch.cat([target_bins, bins], 1)))
        alternatives = (F.adaptive_avg_pool2d(analysis, (4, 4)), bins,
                        reference_tokens(projected), grids, engine.condition(context, analysis))
        for mode, vector in zip(CONDITIONS, alternatives, strict=True):
            if not torch.isfinite(vector).all():
                raise ValueError(f"Invalid feature at {mode}: {donor}/{tag}")
            output[mode].append(vector.flatten().cpu().numpy())
    return {mode: np.stack(rows) for mode, rows in output.items()}


def labels(family, paired):
    directions, unique = {}, set()
    for scene in FAMILIES[family]:
        for direction in DIRECTIONS[scene]:
            a, b = direction.split("-to-")
            anchor = f"{a}-off" if paired else None
            lit, edited = (scene, f"{b}-off", anchor, -1), (scene, f"{a}-on", anchor, 1)
            directions[f"{scene}/{direction}"] = lit, edited
            unique.update((lit, edited))
    return directions, sorted(unique)


def vector_matrix(lookup, rows):
    vectors = []
    for scene, picture, anchor, _ in rows:
        value = lookup[scene, picture].astype(np.float64)
        if anchor is not None:
            value -= lookup[scene, anchor].astype(np.float64)
            length = np.linalg.norm(value)
            if length:
                value /= length
        vectors.append(value)
    return np.stack(vectors)


def train_readout(lookup, rows):
    x = vector_matrix(lookup, rows)
    target = np.array([row[-1] for row in rows])
    counts = {(scene, label): sum(row[0] == scene and row[-1] == label for row in rows)
              for scene, _, _, label in rows}
    if len(counts) != 4:
        raise ValueError("Expected both outcomes in both training scenes")
    importance = np.array([.25 / counts[row[0], row[-1]] for row in rows])
    origin = importance @ x
    centered = x - origin
    scale = np.sqrt(importance @ np.sum(centered**2, axis=1))
    if not scale:
        return origin, 1., np.zeros(x.shape[1])
    weights = np.sqrt(importance)
    geometry = centered / scale * weights[:, None]
    gram = geometry @ geometry.T
    coefficients = np.linalg.solve(gram + REGULARIZATION * np.eye(len(rows)), weights * target)
    return origin, scale, geometry.T @ coefficients


def evaluate(lookup, train, test, paired):
    _, fit = labels(train, paired)
    directions, unseen = labels(test, paired)
    origin, scale, classifier = train_readout(lookup, fit)
    estimates = (vector_matrix(lookup, unseen) - origin) @ classifier / scale
    predictions = dict(zip(unseen, estimates, strict=True))
    values = {}
    for key, (lit, edited) in directions.items():
        margin = float(predictions[edited] - predictions[lit])
        comparison = "tie" if abs(margin) <= 1e-12 else ("win" if margin > 0 else "loss")
        values[key] = dict(margin=margin, physical=float(predictions[lit]), editor=float(predictions[edited]), outcome=comparison)
    counts = {kind: sum(row["outcome"] == kind for row in values.values()) for kind in ("win", "loss", "tie")}
    good = {}
    by_class = {}
    for category in (-1, 1):
        name = "physical" if category == -1 else "editor"
        selected = [float(score * category) for row, score in predictions.items() if row[-1] == category]
        good[name] = sum(value > 1e-12 for value in selected) / len(selected)
        by_class[name] = dict(total=len(selected), correct=sum(value > 1e-12 for value in selected))
    balanced = sum(good.values()) / len(good)
    passes = counts["win"] * 3 >= 2 * len(values) and balanced >= (2/3)
    return dict(unique_train=len(fit), class_counts=by_class, balanced_accuracy=balanced, directions=values,
                rankings=counts, passed=passes)


def run(base, out):
    torch.set_num_threads(4)
    code = Path(__file__)
    if digest(code.with_name("reference-signal-readout-protocol.md")) != PROTOCOL:
        raise ValueError("Changed readout protocol")
    parent = base / "positional-lightness-training-v1"
    receipt_path = parent / "training.json"
    if digest(receipt_path) != TRAINING:
        raise ValueError("Changed training receipt")
    parent_info = json.loads(receipt_path.read_bytes())
    out.mkdir()
    pictures, target = read_frames(frozen_scenes(base))
    locations = sorted(pictures)
    ordered = {key: pictures[key] for key in locations}
    current = SpatialEngine("multiref-v1")
    result, evidence = {}, {}
    for train, test in (tuple(FAMILIES), tuple(reversed(FAMILIES))):
        if set(FAMILIES[train]) & set(FAMILIES[test]):
            raise ValueError("Source-family overlap")
        model = ConditionedEngine(True)
        weight = parent / train / "local.pth"
        info = parent_info["folds"][train]["local"]
        if digest(weight) != info["head_sha256"] or model.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed positional head")
        model.head.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        extracted = features(model, current, ordered, target)
        modes = {}
        feature_path = out / f"train-{train}.npz"
        np.savez_compressed(feature_path, **extracted)
        for name, matrix in extracted.items():
            lookup = dict(zip(locations, matrix, strict=True))
            modes[name] = {key: evaluate(lookup, train, test, paired)
                           for key, paired in (("reference", False), ("known_anchor", True))}
        result[test] = dict(trained_on=train, modes=modes)
        evidence[train] = dict(sha256=digest(feature_path), weight=info["head_sha256"],
                               arrays={name: sha256(value.tobytes()).hexdigest() for name, value in extracted.items()})
        print(test, {name: {k: (d["rankings"], round(d["balanced_accuracy"], 3)) for k, d in mode.items()}
                     for name, mode in modes.items()}, flush=True)
    both = {name: {mode: all(row["modes"][name][mode]["passed"] for row in result.values())
                   for mode in ("reference", "known_anchor")} for name in CONDITIONS}
    json_to(out / "results.json", dict(checkpoint=CHECKPOINT, protocol=PROTOCOL, script=digest(code),
                                      training=TRAINING, image_rows=locations, features=evidence,
                                      results=result, both_folds=both))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
