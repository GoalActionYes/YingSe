"""Read known style response from two-image renders with a fixed nuisance projection."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from probe_capture_invariant_look import render_style, style_transforms
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

STYLES = ("film_a", "film_b")
STATES = ("off", *STYLES)


def remove_gain(delta, sensitivity):
    axes = (0, 1)
    factor = np.sum(delta * sensitivity, axis=axes) / np.sum(sensitivity**2, axis=axes)
    return delta - sensitivity * factor


def bounded_fit(templates, observation):
    # The feasible set is a triangle. Compare the unconstrained solution and the three edges.
    gram = templates.T @ templates
    cross = templates.T @ observation
    solution = np.linalg.solve(gram, cross)
    if (solution >= 0).all() and solution.sum() <= 1:
        return solution
    first = float(np.clip(cross[0] / gram[0, 0], 0, 1))
    second = float(np.clip(cross[1] / gram[1, 1], 0, 1))
    denominator = gram[0, 0] + gram[1, 1] - 2 * gram[0, 1]
    diagonal = float(np.clip((cross[0] - cross[1] - gram[0, 1] + gram[1, 1]) / denominator, 0, 1))
    candidates = ([first, 0], [0, second], [diagonal, 1 - diagonal])
    return np.asarray(min(candidates, key=lambda w: w @ gram @ w - 2 * (w @ cross)))


def grade(engine, target, reference):
    return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)


def projections(target, reference, model_output, self_output, edits, gain):
    residual = model_output.astype(np.float64) - self_output
    templates = np.column_stack([edit.reshape(-1) for edit in edits])
    cleaned = np.column_stack([remove_gain(edit, gain).reshape(-1) for edit in edits])
    results = {}
    for arm, design, signal in (
        ("direct", templates, residual), ("subtract_gain", cleaned, remove_gain(residual, gain)),
    ):
        if np.array_equal(target, reference):
            outputs, weights = target.copy(), np.zeros(2)
        else:
            weights = bounded_fit(design, signal.reshape(-1))
            outputs = np.rint(np.clip(target + (templates @ weights).reshape(target.shape), 0, 255)).astype(np.uint8)
        results[arm] = outputs, weights.tolist()
    kept = (np.linalg.norm(cleaned, axis=0) / np.linalg.norm(templates, axis=0)).tolist()
    return results, kept


def controls():
    identity = np.eye(2)
    np.testing.assert_allclose(bounded_fit(identity, np.array([.25, .5])), [.25, .5])
    np.testing.assert_allclose(bounded_fit(identity, np.array([1., 1.])), [.5, .5])
    np.testing.assert_allclose(bounded_fit(identity, np.array([-1., .5])), [0., .5])
    gain = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    np.testing.assert_allclose(remove_gain(gain * [2, 3, 4], gain), 0)


def run(out):
    controls()
    source = BASE / "gros-film-b-cross-v2"
    frozen = checked_json(source / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    ops = style_transforms(BASE)
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed model or strength")
    photos = {}
    for key, record in frozen["inputs"].items():
        original, authored = [
            checked_rgb(source / f"{key}-{label}.png", record["hashes"][label]) for label in ("off", "on")
        ]
        if not np.array_equal(render_style(original, ops["film_b"]), authored):
            raise ValueError("Known Film B render disagrees with its archived reference")
        photos[key] = {"off": original, "film_a": render_style(original, ops["film_a"]), "film_b": authored}

    out.mkdir()
    cases = {}
    for person in GROUPS:
        target = photos[f"{person}-cold"]["off"]
        expected = {style: finish_spatial(target, render_style(target, ops[style]), STRENGTH) for style in STYLES}
        edits = [expected[style].astype(np.float64) - target for style in STYLES]
        value = target.astype(np.float64) / 255
        gain = np.where(value <= .04045, value, (value + .055) / 2.4) * 255
        self_output = grade(engine, target, target)
        same, kept = projections(target, target, self_output, self_output, edits, gain)
        if not all(np.array_equal(pair[0], target) for pair in same.values()):
            raise ValueError("Same-image control changed")
        area = masks(labels[person]["cold"], labels["screen"], target.shape[:2])
        for donor in GROUPS:
            if donor == person:
                continue
            name = f"{person}-from-{donor}"
            frames = {arm: {light: {} for light in LIGHTS} for arm in ("frozen13", "direct", "subtract_gain")}
            weights, hashes = {}, {}
            for light in LIGHTS:
                for state in STATES:
                    reference = photos[f"{donor}-{light}"][state]
                    rendered = grade(engine, target, reference)
                    frames["frozen13"][light][state] = rendered
                    if state in ("off", "film_b"):
                        prior = f"{light}-{'off' if state == 'off' else 'on'}"
                        recorded = checked_rgb(source / f"{name}-13-{prior}.png", frozen["cases"][name]["hashes"][f"13-{prior}"])
                        if not np.array_equal(recorded, rendered):
                            raise ValueError("Frozen 13 baseline changed")
                    inferred, _ = projections(target, reference, rendered, self_output, edits, gain)
                    for arm, (result, coefficient) in inferred.items():
                        frames[arm][light][state] = result
                        key = f"{name}-{arm}-{light}-{state}"
                        weights[key] = coefficient
                        hashes[key] = save(out / f"{key}.png", result)
            scores = {arm: {style: score(target, expected[style], {
                light: {"off": values["off"], "on": values[style]} for light, values in states.items()
            }, area) for style in STYLES} for arm, states in frames.items()}
            board = panel(out / f"{name}-comparison.png", {"target": {"off": target, **expected}} | {
                f"{arm} {light}": values for arm, states in frames.items() for light, values in states.items()
            })
            cases[name] = dict(scores=scores, weights=weights, hashes=hashes, board=board, remaining_template_norm=kept)
    protocol = Path(__file__).with_name("gros-render-intent-projection-protocol.md")
    json_to(out / "results.json", dict(parent=BASELINE_SHA, regions=REGIONS_SHA, checkpoint=CHECKPOINT,
                                       strength=STRENGTH, protocol=digest(protocol), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
