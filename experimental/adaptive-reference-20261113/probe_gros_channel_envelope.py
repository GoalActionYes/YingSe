"""Bound the portrait colors available from frozen two-image outputs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGRESSION_SHA, REGIONS_SHA, STATES
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save

SHARED_SHA = "f8174f15633023e4e7295c88d0bc892adebe44949865d4a37a930b8659081654"
METHODS = ("stable", "13", "reference", "two_image", "shared", "equal", "identity")


def interval_distance(value, low, high):
    return np.maximum(low - value, 0) + np.maximum(value - high, 0)


def best_change(weight, capacity, budget):
    order = np.argsort(-weight, axis=None, kind="stable")
    span = capacity.ravel()[order]
    spend = np.clip(budget - np.cumsum(span) + span, 0, span)
    return float(spend @ weight.ravel()[order])


def necessary(target, truth, bounds, where):
    src, goal = (img[where].astype(np.int32) for img in (target, truth))
    signal = goal - src
    weight = np.abs(signal)
    power = np.square(signal.astype(np.float64)).sum()
    budget = 3 * signal.size
    converted = {light: {state: (lo[where].astype(np.int32), hi[where].astype(np.int32))
                         for state, (lo, hi) in rows.items()} for light, rows in bounds.items()}
    light_checks = {}
    for light, rows in converted.items():
        low_off, high_off = rows["off"]
        low_on, high_on = rows["on"]
        if np.any((src < low_off) | (src > high_off)):
            raise ValueError("Original should be an allowed off state")
        original_error = interval_distance(goal, low_on, high_on).sum()
        free_min = interval_distance(signal, low_on - high_off, high_on - low_off).sum()
        least_error = max(original_error - budget, free_min) / signal.size
        if power:
            sign = signal >= 0
            best_on = np.where(sign, high_on, low_on)
            worst_on = np.where(sign, low_on, high_on)
            positive = best_change(weight, np.where(sign, src-low_off, high_off-src), budget)
            negative = best_change(weight, np.where(sign, high_off-src, src-low_off), budget)
            projection = [
                float(((worst_on-src).astype(np.float64).ravel() @ signal.ravel() - negative) / power),
                float(((best_on-src).astype(np.float64).ravel() @ signal.ravel() + positive) / power),
            ]
        else:
            projection = None
        light_checks[light] = dict(response_mae_min=float(least_error), projection=projection,
                                   identity_off_response_mae_min=float(original_error / signal.size))
    spill = {}
    for state in STATES:
        a, b = converted["cold"][state], converted["warm"][state]
        spill[state] = float(np.maximum(0, np.maximum(a[0] - b[1], b[0] - a[1])).mean())
    ruled_out = (any(value > 3 for value in spill.values())
                 or any(row["response_mae_min"] > 3 or row["projection"] is None
                        or row["projection"][0] > 1.25 or row["projection"][1] < .75
                        for row in light_checks.values()))
    return dict(lighting=light_checks, spill_mae_min=spill, ruled_out=ruled_out)


def choices(target, expected, variants):
    limits = {light: {state: (stack.min(0), stack.max(0)) for state, stack in states.items()}
              for light, states in variants.items()}
    chosen, clipped = {}, {}
    y, x = np.indices(target.shape[:2], sparse=True)
    for light, states in variants.items():
        images = states["on"]
        errors = np.abs(images.astype(np.int16) - expected).sum(axis=-1)
        on = images[np.argmin(errors, axis=0), y, x]
        low, high = limits[light]["on"]
        nearest = np.clip(expected, low, high)
        if np.any(np.abs(nearest.astype(np.int16)-expected) > np.abs(on.astype(np.int16)-expected)):
            raise ValueError("Relaxed color box is unexpectedly narrower than an endpoint")
        chosen[light] = {"off": target, "on": on}
        clipped[light] = {"off": target, "on": nearest}
    return limits, {"endpoint": chosen, "box": clipped}


def sources(root, baseline, regression, previous, case, target):
    prior = BASE / "gros-two-image-edit-regression-v1"
    candidates = {}
    for light in LIGHTS:
        candidates[light] = {}
        for state in STATES:
            results = []
            for arm in METHODS[:4]:
                primary = arm in ("stable", "13")
                role = state if primary or state == "off" else "film_b"
                key = f"{arm}-{light}-{role}"
                folder, ledger = (root, baseline) if primary else (prior, regression)
                results.append(checked_rgb(folder / f"{case}-{key}.png", ledger["hashes"][key]))
            for arm in METHODS[4:-1]:
                key = f"{light}-{state}"
                folder = BASE / "shared-chroma-direction-v1" / "portrait"
                results.append(checked_rgb(folder / f"{case}-{arm}-{key}.png", previous[arm]["images"][key]["sha256"]))
            candidates[light][state] = np.stack([*results, target])
    return candidates


def run(out):
    np.testing.assert_array_equal(interval_distance(np.array([-2, 1, 6]), 0, 4), [2, 0, 2])
    assert best_change(np.array([3, 2, 1]), np.array([1, 3, 4]), 2) == 5
    root = BASE / "gros-film-b-cross-v2"
    frozen = checked_json(root / "results.json", BASELINE_SHA)
    regression = checked_json(BASE / "gros-two-image-edit-regression-v1" / "results.json", REGRESSION_SHA)
    previous = checked_json(BASE / "shared-chroma-direction-v1" / "results.json", SHARED_SHA)
    groups = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    if regression["parent"] != BASELINE_SHA or previous["portrait_baseline"] != BASELINE_SHA:
        raise ValueError("Source experiment did not use frozen portrait cases")
    out.mkdir()
    cases = {}
    for case, baseline in frozen["cases"].items():
        name = case.split("-from-")[0]
        src = checked_rgb(root / f"{name}-cold-off.png", frozen["inputs"][f"{name}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{case}-expected.png", baseline["expected_sha"])
        zones = masks(groups[name]["cold"], groups["screen"], src.shape[:2])
        variants = sources(root, baseline, regression["cases"][case], previous["portrait"][case], case, src)
        limits, witnesses = choices(src, expected, variants)
        decisions = {name: necessary(src, expected, limits, area) for name, area in zones.items()}
        views = {"target": {"original": src, "Film B": expected}}
        diagnostics = {}
        for arm, states in witnesses.items():
            scores = score(src, expected, states, zones)
            if any(scores[zone]["passed"] and row["ruled_out"] for zone, row in decisions.items()):
                raise ValueError("Witness contradicted the necessary condition")
            artifacts = {}
            for light, photos in states.items():
                views[f"{arm} {light}"] = photos
                artifacts[light] = {state: save(out / f"{case}-{arm}-{light}-{state}.png", img)
                                    for state, img in photos.items()}
            diagnostics[arm] = dict(scores=scores, artifacts=artifacts)
        cases[case] = dict(necessary=decisions, diagnostics=diagnostics,
                          board=panel(out / f"{case}-comparison.png", views))
    protocol = Path(__file__).with_name("gros-channel-envelope-protocol.md")
    json_to(out / "results.json", dict(parent=BASELINE_SHA, regression=REGRESSION_SHA, shared=SHARED_SHA,
                                       protocol=digest(protocol), candidates=METHODS, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
