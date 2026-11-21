"""Project the expected RGB to the convex hull of frozen portrait renders."""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np

from probe_gros_channel_envelope import (
    BASELINE_SHA, METHODS, REGIONS_SHA, REGRESSION_SHA, SHARED_SHA, sources,
)
from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save

PARENT = "38d359707d59781bf5598f16e8ef53bdacbc2d31c6c272339b61c98229586d07"


def dot(a, b):
    return np.sum(a * b, axis=-1)


def barycentric_target(vertices, goal):
    a, b, c, d = vertices
    u, v, w, p = b-a, c-a, d-a, goal-a
    vw, wp, pw = np.cross(v, w), np.cross(w, u), np.cross(u, v)
    determinant = dot(u, vw)
    numerators = np.stack((dot(p, vw), dot(p, wp), dot(p, pw)), axis=-1)
    weights = np.divide(numerators, determinant[:, None], out=np.zeros_like(numerators),
                        where=determinant[:, None] != 0)
    return (determinant != 0) & (weights.min(axis=-1) >= 0) & (weights.sum(axis=-1) <= 1)


def project_chunk(variants, goal):
    loss = np.square(variants - goal).sum(axis=-1)
    j = np.arange(len(goal))
    index = loss.argmin(axis=0)
    nearest = variants[index, j].copy()
    best = loss[index, j]

    def consider(candidate, valid=True):
        cost = np.square(candidate - goal).sum(axis=-1)
        replace = valid & (cost < best)
        nearest[replace] = candidate[replace]
        best[replace] = cost[replace]

    for a, b in combinations(variants, 2):
        u = b-a
        norm = dot(u, u)
        t = np.divide(dot(goal-a, u), norm, out=np.zeros_like(norm), where=norm != 0)
        consider(a + np.clip(t, 0, 1)[:, None] * u)
    for a, b, c in combinations(variants, 3):
        u, v, p = b-a, c-a, goal-a
        uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
        up, vp = dot(u, p), dot(v, p)
        det = uu*vv - uv*uv
        q = np.divide(vv*up-uv*vp, det, out=np.zeros_like(det), where=det > 0)
        r = np.divide(uu*vp-uv*up, det, out=np.zeros_like(det), where=det > 0)
        consider(a + q[:, None]*u + r[:, None]*v, (det > 0) & (q >= 0) & (r >= 0) & (q+r <= 1))
    for indexes in combinations(range(len(variants)), 4):
        unresolved = best > 0
        if not unresolved.any():
            break
        inside = barycentric_target([variants[i, unresolved] for i in indexes], goal[unresolved])
        use = np.flatnonzero(unresolved)[inside]
        nearest[use] = goal[use]
        best[use] = 0
    return nearest


def project(variants, expected):
    stack = variants.reshape(len(variants), -1, 3)
    wanted = expected.reshape(-1, 3)
    result = np.empty(wanted.shape, np.float64)
    for lower in range(0, len(wanted), 32768):
        take = slice(lower, lower+32768)
        result[take] = project_chunk(stack[:, take].astype(np.float64), wanted[take].astype(np.float64))
    return result.reshape(expected.shape)


def certificates(variants, target, expected, chosen):
    signal = expected.astype(np.float64) - target
    direction = expected - chosen
    norm = abs(direction).max(axis=-1, keepdims=True)
    unit = np.divide(direction, norm, out=np.zeros_like(direction), where=norm > 0)
    magnitude = np.abs(unit).sum(axis=-1)
    lower = np.maximum(0, dot(unit, expected) - dot(unit, variants).max(axis=0) - .5*magnitude)
    power = np.square(signal).sum(axis=-1)
    overlap = dot(signal, variants.astype(np.float64) - target)
    margin = .5 * np.abs(signal).sum(axis=-1)
    return lower, power, overlap.min(axis=0) - margin, overlap.max(axis=0) + margin


def bounds(stats, area):
    loss, power, minimum, maximum = [value[area] for value in stats]
    total = power.sum()
    interval = [float(minimum.sum() / total), float(maximum.sum() / total)] if total else None
    mae = float(loss.mean() / 3)
    excluded = mae > 3 or interval is None or interval[0] > 1.25 or interval[1] < .75
    return dict(response_mae_lower=mae, projection_interval=interval, blocked_with_exact_off=excluded)


def run(out):
    basis = np.array([[[0., 0., 0.]], [[6., 0., 0.]], [[0., 6., 0.]], [[0., 0., 6.]]])
    for goal, answer in (([1, 2, 3], [1, 2, 3]), ([1, 1, 1], [1, 1, 1]),
                         ([6, 6, 6], [2, 2, 2]), ([4, 4, -4], [3, 3, 0])):
        np.testing.assert_allclose(project_chunk(basis, np.array([goal], float)), [answer], atol=1e-12)
    root = BASE / "gros-film-b-cross-v2"
    frozen = checked_json(root / "results.json", BASELINE_SHA)
    regression = checked_json(BASE / "gros-two-image-edit-regression-v1" / "results.json", REGRESSION_SHA)
    last = checked_json(BASE / "shared-chroma-direction-v1" / "results.json", SHARED_SHA)
    prev_out = BASE / "gros-channel-envelope-v1"
    previous = checked_json(prev_out / "results.json", PARENT)
    if tuple(previous["candidates"]) != METHODS or previous["parent"] != BASELINE_SHA:
        raise ValueError("Changed comparison suite")
    groups = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    out.mkdir()
    cases = {}
    for case, baseline in frozen["cases"].items():
        name = case.split("-from-")[0]
        src = checked_rgb(root / f"{name}-cold-off.png", frozen["inputs"][f"{name}-cold"]["hashes"]["off"])
        expected = checked_rgb(root / f"{case}-expected.png", baseline["expected_sha"])
        zones = masks(groups[name]["cold"], groups["screen"], src.shape[:2])
        roster = sources(root, baseline, regression["cases"][case], last["portrait"][case], case, src)
        anchor = previous["cases"][case]["diagnostics"]
        panel_rows = {"target": {"original": src, "expected": expected}}
        witnesses, reports, checks = {}, {}, {}
        for light in LIGHTS:
            variants = roster[light]["on"]
            mixed = project(variants, expected)
            frame = np.rint(mixed).astype(np.uint8)
            stats = certificates(variants, src, expected, mixed)
            checks[light] = {label: bounds(stats, area) for label, area in zones.items()}
            reports[light] = {state: save(out / f"{case}-{light}-{state}.png", rgb)
                              for state, rgb in (("off", src), ("on", frame))}
            witnesses[light] = {"off": src, "on": frame}
            panel_rows[light] = {arm: checked_rgb(prev_out / f"{case}-{arm}-{light}-on.png",
                                                 record["artifacts"][light]["on"]) for arm, record in anchor.items()}
            panel_rows[light]["RGB mixture"] = frame
        scores = score(src, expected, witnesses, zones)
        for light in LIGHTS:
            for label, row in checks[light].items():
                response = scores[label]["response"][light]
                lo, hi = row["projection_interval"]
                if not (lo-1e-10 <= response["projection"] <= hi+1e-10
                        and row["response_mae_lower"] <= response["response_mae"] + 1e-10):
                    raise ValueError("Witness contradicts conservative bounds")
        cases[case] = dict(scores=scores, bounds=checks, images=reports, board=panel(out / f"{case}-comparison.png", panel_rows))
    protocol = Path(__file__).with_name("gros-rgb-mixture-protocol.md")
    json_to(out / "results.json", dict(parent=PARENT, candidates=METHODS, protocol=digest(protocol), cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
