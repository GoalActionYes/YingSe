"""Measure whether context outside an authored edit predicts normal wall lightness."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_cross_source_wall_lightening_workbench import DIRECTIONS, frozen_scenes, load
from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import LUMA
from probe_shared_authored_tone_admission import support
from probe_target_mirror_chroma import digest, json_to, panel
from train_positional_lightness import FAMILIES


MIN_COVERAGE = .80
LIMIT = 3.
RIDGE = .1
MODES = ("mean", "spatial")


def outside_tiles(shape, extent, measurement):
    height, width = shape
    top = np.linspace(0, height, 9, dtype=int)
    left = np.linspace(0, width, 9, dtype=int)
    y, x = np.ogrid[:height, :width]
    x0, y0, x1, y1 = extent
    excluded = measurement | ((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    allowed = np.zeros(shape, dtype=bool)
    tiles = []
    for v0, v1 in zip(top[:-1], top[1:]):
        for u0, u1 in zip(left[:-1], left[1:]):
            tile = np.s_[v0:v1, u0:u1]
            if not excluded[tile].any():
                allowed[tile] = True
                tiles.append((int(u0), int(v0), int(u1), int(v1)))
    return tiles, allowed


def read_one(rgb, measurement, coords):
    luma = rgb @ LUMA
    usable = support(rgb) & measurement
    coverage = float(usable.sum() / measurement.sum())
    wall = float(np.median(luma[usable])) if coverage >= MIN_COVERAGE else None
    means = [float(luma[y0:y1, x0:x1].mean()) for x0, y0, x1, y1 in coords]
    return dict(coverage=coverage, wall=wall, tiles=means)


def ridge_fit(features, values):
    center = features.mean(axis=0)
    intercept = float(values.mean())
    row_vectors = features - center
    scale = float(np.sqrt(np.square(row_vectors).sum(axis=1).mean()))
    if not scale:
        return center, intercept, np.zeros(features.shape[1]), True
    design = row_vectors / scale
    kernel = design @ design.T
    coeff = np.linalg.solve(kernel + RIDGE * np.eye(len(values)), values - intercept)
    return center, intercept, design.T @ coeff / scale, False


def self_check():
    features = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    values = np.array([20., 40., 10., 30.])
    center, intercept, weights, _ = ridge_fit(features, values)
    estimates = (features-center) @ weights + intercept
    assert np.abs(estimates-values).max() < 1
    pooled = features.mean(axis=1, keepdims=True)
    center, intercept, weights, _ = ridge_fit(pooled, values)
    guesses = (pooled-center) @ weights + intercept
    assert guesses[1] == guesses[2] and values[1] != values[2]


def compare(records, calibrate, cases, mode):
    weights = {key: np.array(row["tiles"], dtype=np.float64) for key, row in records.items()}
    if mode == "mean":
        weights = {name: np.mean(data, keepdims=True) for name, data in weights.items()}
    train = np.stack([weights[name] for name in calibrate])
    truth = np.array([records[name]["wall"] for name in calibrate])
    center, intercept, direction, degenerate = ridge_fit(train, truth)
    result = {}
    for role, tag in cases.items():
        observed = records[tag]["wall"]
        predicted = float((weights[tag] - center) @ direction + intercept)
        result[role] = dict(tag=tag, observed=observed, predicted=predicted, residual=observed-predicted)
    expected_effect = result["edit"]["observed"] - result["anchor"]["observed"]
    inferred_effect = result["edit"]["residual"]
    projection = inferred_effect / expected_effect if expected_effect > 0. else None
    accepted = all(abs(result[k]["residual"]) <= LIMIT for k in ("anchor", "physical"))
    accepted &= (projection is not None and .75 <= projection <= 1.25 and inferred_effect >= LIMIT
                 and abs(inferred_effect - expected_effect) <= LIMIT)
    return dict(states=result, measured_edit=expected_effect, inferred_edit=inferred_effect,
                projection=projection, degenerate=degenerate, quantitative_pass=bool(accepted),
                coefficients=direction.tolist(), train_fit_y_prime=intercept)


def analyze_scene(name, scene, out):
    native_levels = sorted(tag for tag, row in scene["frames"].items() if row["metrics"]["admitted"])
    all_cases = {}
    for direction in DIRECTIONS[name]:
        a, b = direction.split("-to-")
        all_cases[direction] = dict(anchor=f"{a}-off", physical=f"{b}-off", edit=f"{a}-on")
    level_tags = sorted({f"{tag}-off" for tag in native_levels} |
                        {tag for labels in all_cases.values() for tag in labels.values()})
    images, hashes = {}, {}
    for key in level_tags:
        file, images[key] = load(scene, key)
        hashes[key] = digest(file)
    initial = images[level_tags[0]]
    area = box(initial, scene["boxes"]["wall_left"])
    if not area.any() or any(pic.shape != initial.shape for pic in images.values()):
        raise ValueError(f"Invalid frozen geometry: {name}")
    coords, valid = outside_tiles(initial.shape[:2], scene["extent"], area)
    if len(coords) < 8:
        raise ValueError(f"Insufficient context area: {name}")
    records = {tag: dict(read_one(pic, area, coords), sha256=hashes[tag]) for tag, pic in images.items()}
    results = {}
    for direction, states in all_cases.items():
        a, b = direction.split("-to-")
        selected = [f"{level}-off" for level in native_levels if level not in (a, b)
                    and records[f"{level}-off"]["wall"] is not None]
        same_context = bool(np.array_equal(images[states["anchor"]][valid], images[states["edit"]][valid]))
        held = all(records[tag]["wall"] is not None for tag in states.values())
        allowed = len(selected) >= 2 and same_context and held
        models = {mode: compare(records, selected, states, mode) for mode in MODES} if allowed else {}
        results[direction] = dict(train=selected, test=states, exact_context=same_context, valid_wall=held,
                                  passed={mode: allowed and models[mode]["quantitative_pass"] for mode in MODES},
                                  models=models)
    boards = {direction: panel(out / f"{name}-{direction}-context.png", {
        "reference": {key: images[tag] for key, tag in states.items()},
        "context inputs": {key: np.where(valid[:, :, None], images[tag], 20)
                           for key, tag in states.items()},
    }) for direction, states in all_cases.items()}
    print(name, {mode: sum(r["passed"][mode] for r in results.values()) for mode in MODES}, flush=True)
    return dict(directions=results, tiles=coords, context_fraction=float(valid.mean()),
                fixed_measurement_bounds=scene["boxes"]["wall_left"], images=records, boards=boards)


def run(base, out):
    self_check()
    frozen = frozen_scenes(base)
    out.mkdir()
    rows = {name: analyze_scene(name, scene, out) for name, scene in frozen.items()}
    families = {}
    for family, names in FAMILIES.items():
        entries = {f"{name}/{direction}": row["passed"] for name in names
                   for direction, row in rows[name]["directions"].items()}
        passing = {mode: [label for label, row in entries.items() if row[mode]] for mode in MODES}
        new = sorted(set(passing["spatial"]) - set(passing["mean"]))
        lost = sorted(set(passing["mean"]) - set(passing["spatial"]))
        admitted = len(passing["spatial"]) * 3 >= 2 * len(entries) and bool(new) and not lost
        families[family] = dict(total=len(entries), passed=passing, new=new, lost=lost, admitted=admitted)
    code = Path(__file__)
    json_to(out/"results.json", dict(scenes=rows, families=families, ridge=RIDGE,
            maximum_error=LIMIT, minimum_coverage=MIN_COVERAGE,
            admitted=all(row["admitted"] for row in families.values()), script=digest(code),
            protocol=digest(code.with_name("reference-context-light-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
