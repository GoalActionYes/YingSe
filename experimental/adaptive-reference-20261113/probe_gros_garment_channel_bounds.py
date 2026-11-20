"""Use conservative color-ratio intervals when clothing channels hit a limit."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from probe_gros_garment_capture_cue import estimate, run
from probe_physical_blue_global_fit import linear
from probe_target_mirror_chroma import checked_json, digest


BASELINE_SHA = '549ecfb4d48573b384563038a7657465ddc8ae3e444e9a96051fc5f82676bc56'


def reportable(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def log_bounds():
    codes = np.arange(256, dtype=float)
    with np.errstate(divide='ignore'):
        lower = np.log(linear(np.maximum(codes - .5, 0)))
        upper = np.log(linear(np.minimum(codes + .5, 255)))
    lower[:8] = -np.inf
    upper[248:] = np.inf
    return lower, upper


LOG_LOWER, LOG_UPPER = log_bounds()


def ratios(pixels):
    red_blue = pixels[:, [0, 2]]
    green = pixels[:, 1:2]
    lower = LOG_LOWER[red_blue] - LOG_UPPER[green]
    upper = LOG_UPPER[red_blue] - LOG_LOWER[green]
    a, b = (len(pixels) - 1) // 2, len(pixels) // 2
    floor = np.partition(lower, a, axis=0)[a]
    ceiling = np.partition(upper, b, axis=0)[b]
    evidence = dict(pixels=len(pixels), finite_fraction=(np.isfinite(lower) & np.isfinite(upper)).mean(axis=0).tolist(),
                    median_lower=reportable(floor), median_upper=reportable(ceiling))
    return floor, ceiling, evidence


def interval_geometry(lower, upper):
    finite_lower = np.where(np.isfinite(lower), lower, np.where(np.isfinite(upper), upper, 0))
    finite_upper = np.where(np.isfinite(upper), upper, finite_lower)
    corners = np.array(list(product(*zip(finite_lower, finite_upper))))
    rays = np.concatenate((-np.eye(2)[~np.isfinite(lower)], np.eye(2)[~np.isfinite(upper)]))
    closest = np.clip(np.zeros(2), lower, upper)
    minimum = float(np.linalg.norm(closest))
    maximum = None if len(rays) else float(np.linalg.norm(corners, axis=1).max())
    directions = np.concatenate((corners, rays))
    return minimum, maximum, directions


def classify(lower, upper, facial_residual):
    minimum, maximum, points = interval_geometry(lower, upper)
    face_size = np.linalg.norm(facial_residual)
    if maximum is not None and maximum < .10:
        return ('local' if face_size >= .10 else 'global'), minimum, maximum
    if minimum >= .10 and face_size >= .10:
        projected = points @ (facial_residual / face_size)
        if np.all(projected >= .50 * np.linalg.norm(points, axis=1)):
            return 'capture', minimum, maximum
    return 'uncertain', minimum, maximum


def estimate_bounds(before, after, before_zones, after_zones):
    original = estimate(before, after, before_zones, after_zones)
    changes = original['changes']
    background, face, shirt = (changes[zone]['delta'] for zone in ('background', 'face', 'shirt'))
    if background is None or face is None or shirt is not None:
        return original
    original_shirt = before[before_zones['shirt']]
    altered_shirt = after[after_zones['shirt']]
    if min(len(original_shirt), len(altered_shirt)) < 256:
        return original
    old_low, old_high, prior = ratios(original_shirt)
    new_low, new_high, current = ratios(altered_shirt)
    minimum = new_low - old_high - background
    maximum = new_high - old_low - background
    face_residual = np.array(face) - background
    route, min_norm, max_norm = classify(minimum, maximum, face_residual)
    return {**original, 'route': route, 'face_residual': face_residual.tolist(),
            'bounds': dict(before=prior, after=current, residual_lower=reportable(minimum),
                           residual_upper=reportable(maximum), min_norm=min_norm, max_norm=max_norm)}


def study(gros, scope, baseline, out):
    archived = checked_json(baseline / 'results.json', BASELINE_SHA)
    run(gros, scope, out, estimate_material=estimate_bounds,
        additions={'prior_garment_sha256': BASELINE_SHA, 'bounds_script_sha256': digest(Path(__file__))})
    completed = json.loads((out / 'results.json').read_bytes())
    if completed['author'] != archived['author']:
        raise ValueError('Changed real author fixture')
    for group, states in archived['donor_measurements'].items():
        for state, original in states.items():
            current = completed['donor_measurements'][group][state]
            if original['changes']['shirt']['delta'] is not None and original != current:
                raise ValueError('Changed a known clothing state')
    for key, case in archived['cases'].items():
        _, donor = key.split('-from-')
        old, new = case['arms'], completed['cases'][key]['arms']
        if old['background'] != new['background']:
            raise ValueError('Changed frozen background outputs')
        for state, sha in old['garment']['hashes'].items():
            previous = archived['donor_measurements'][donor][state]['route']
            current = completed['donor_measurements'][donor][state]['route']
            if previous == current and new['garment']['hashes'][state] != sha:
                raise ValueError('Changed output for a fixed decision')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gros', 'scope', 'baseline', 'out'):
        parser.add_argument(name, type=Path)
    study(**vars(parser.parse_args()))
