"""Check whether shared chroma directions help more than equal attenuation."""

from argparse import ArgumentParser
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA_INTEGER, STRENGTH, SpatialEngine, background,
    checked_json, checked_rgb, digest, facd, json_to, panel, save,
)
from reference_color_app.features import extract_features
from reference_color_app.lut import apply_mapping, build_mapping
from reference_color_app.pipeline import algorithm_fingerprint

STATES = ("off", "on")
ARMS = ("shared", "equal")


def shared_delta(spatial_delta, stable_delta):
    aligned = np.sign(spatial_delta) == np.sign(stable_delta)
    return np.sign(spatial_delta) * np.minimum(abs(spatial_delta), abs(stable_delta)) * aligned


def grade(target, reference, baseline, *, mode):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True, "max_luma_error": 0.}
    mapping = build_mapping(extract_features(target, 0), extract_features(reference, 1))
    stable = apply_mapping(target, mapping, STRENGTH)
    luma = baseline @ LUMA
    original = chroma(target, target @ LUMA)
    spatial_delta = chroma(baseline, luma) - original
    stable_delta = chroma(stable, stable @ LUMA) - original
    shared = shared_delta(spatial_delta, stable_delta)
    energy = float(np.square(shared).sum())
    full_energy = float(np.square(spatial_delta).sum())
    scale = (energy / full_energy)**.5 if full_energy else 0.
    equal = spatial_delta * scale
    np.testing.assert_allclose(np.square(equal).sum(), energy, rtol=1e-12, atol=1e-9)
    result, limited = render(original + (shared if mode == "shared" else equal), luma)
    luma_error = float(np.abs((result.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000)
    if luma_error > .5 + 1e-10:
        raise ValueError("Rendered luminance moved beyond rounding tolerance")
    return result, dict(prelimit_scale=scale, gamut_limited_fraction=limited, max_luma_error=luma_error)


def portraits(out):
    out.mkdir()
    root = BASE / "gros-film-b-cross-v2"
    parent = checked_json(root / "results.json", BASELINE_SHA)
    if parent["strength"] != STRENGTH or parent["spatial"] != CHECKPOINT:
        raise ValueError("Changed portrait renderer")
    zones = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)

    @lru_cache(maxsize=12)
    def reference(person, light, state):
        key = f"{person}-{light}"
        return checked_rgb(root / f"{key}-{state}.png", parent["inputs"][key]["hashes"][state])

    cases = {}
    for name, locked in parent["cases"].items():
        target_id, donor = name.split("-from-")
        origin = reference(target_id, "cold", "off")
        expected = checked_rgb(root / f"{name}-expected.png", locked["expected_sha"])
        area = masks(zones[target_id]["cold"], zones["screen"], origin.shape[:2])
        baseline = {light: {state: checked_rgb(root / f"{name}-13-{light}-{state}.png",
                                               locked["hashes"][f"13-{light}-{state}"]) for state in STATES}
                    for light in LIGHTS}
        base_scores = score(origin, expected, baseline, area)
        if base_scores != locked["scores"]["13"]:
            raise ValueError("Changed portrait baseline scores")
        case = {"baseline": base_scores}
        board = {"target": {"origin": origin, "expected": expected}}
        for light, states in baseline.items():
            board[f"13 {light}"] = states
        for arm in ARMS:
            outputs, hashes = {}, {}
            for light in LIGHTS:
                outputs[light] = {}
                for state in STATES:
                    result, diagnostics = grade(origin, reference(donor, light, state), baseline[light][state], mode=arm)
                    outputs[light][state] = result
                    key = f"{light}-{state}"
                    hashes[key] = diagnostics | {"sha256": save(out / f"{name}-{arm}-{key}.png", result)}
                board[f"{arm} {light}"] = outputs[light]
            scores = score(origin, expected, outputs, area)
            case[arm] = dict(scores=scores, passed=all(row["passed"] for row in scores.values()), images=hashes)
        case["board"] = panel(out / f"{name}-comparison.png", board)
        cases[name] = case
    return cases


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 renderer")
    lhs = np.array([[-4, 3], [2, -6], [3, 0]])
    rhs = np.array([[-1, -10], [6, -4], [0, 3]])
    np.testing.assert_array_equal(shared_delta(lhs, rhs), [[-1, 0], [2, -4], [0, 0]])
    toy = np.array([[[0, 40, 80], [127, 130, 200], [255, 255, 255]]], dtype=np.uint8)
    for arm in ARMS:
        np.testing.assert_array_equal(grade(toy, toy.copy(), 255 - toy, mode=arm)[0], toy)
    out.mkdir()
    people = portraits(out / "portrait")
    checks = {}
    base_cases = {name: run for name, run in (("background", background), ("facd", facd))}
    for arm in ARMS:
        folder = out / arm
        folder.mkdir()
        grade_arm = partial(lambda _, target, ref, frame, mode: grade(target, ref, frame, mode=mode), mode=arm)
        checks[arm] = {name: suite(engine, folder / name, grade_arm, arm) for name, suite in base_cases.items()}
    passes = {}
    for arm in ARMS:
        bg = checks[arm]["background"]
        filters = checks[arm]["facd"]
        passes[arm] = {
            "portrait": [key for key, row in people.items() if row[arm]["passed"]],
            "background": [key for key, row in bg["cases"].items() if row["candidate"]["passed"]],
            "facd": [key for key, row in filters["cases"].items() if all(row["candidate"]["gates"].values())],
        }
    keeps = all(set(passes["equal"][suite]) <= set(passes["shared"][suite]) for suite in passes["shared"])
    improves = any(len(passes["equal"][suite]) < len(passes["shared"][suite]) for suite in passes["shared"])
    passed = (len(passes["shared"]["portrait"]) >= 5 and checks["shared"]["background"]["passed"]
              and checks["shared"]["facd"]["passed"] and keeps and improves)
    protocol = Path(__file__).with_name("shared-chroma-direction-protocol.md")
    json_to(out / "results.json", dict(portrait_baseline=BASELINE_SHA, spatial=CHECKPOINT,
                                       stable=algorithm_fingerprint(), strength=STRENGTH, protocol=digest(protocol),
                                       portrait=people, checks=checks, passes=passes, qualified=passed))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
