"""Test the attainable blend of existing two-image portrait outputs offline."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, save


BASELINE_SHA = '4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260'
REGRESSION_SHA = '6868a57951ec91fd0408fafe1b70a049719e355e4f0ba9a40b026eff01514dd0'
REGIONS_SHA = 'd69083602c80e65a859dd90531861cd283dffe99de90fb878c981d58315235c5'
CANDIDATES = ('stable', '13', 'reference', 'two_image', 'identity')
STATES = ('off', 'on')


def inequality_system(frames, target, truth, zones):
    rows, bounds = [], []

    def add(row, bound):
        rows.append(row)
        bounds.append(bound)

    def bound_mae(residual, maximum):
        for sample in residual:
            direction = np.sign(sample)
            row = (residual * direction).mean(axis=(1, 2))
            add(row, maximum)
            add(-row, maximum)

    for mask in zones.values():
        source = target[mask].astype(np.float64)
        signal = truth[mask].astype(np.float64) - source
        power = np.square(signal).sum()
        if power == 0:
            raise ValueError('Missing authored style signal')
        rounding = np.abs(signal).sum() / power
        for light in LIGHTS:
            off, on = (frames[light][state][:, mask].astype(np.float64) for state in STATES)
            response = on - off
            projection = (response * signal).sum(axis=(1, 2)) / power
            add(projection, 1.25 + rounding)
            add(-projection, -.75 + rounding)
            bound_mae(off - source, 3.5)
            bound_mae(response - signal, 4.)
        for state in STATES:
            delta = frames['cold'][state][:, mask].astype(np.float64) - frames['warm'][state][:, mask]
            bound_mae(delta, 4.)
    return np.stack(rows), np.array(bounds)


def solve(rows, bounds):
    count = len(CANDIDATES)
    result = linprog(
        np.r_[np.zeros(count), 1.],
        A_ub=np.column_stack((rows, -np.ones(len(rows)))),
        b_ub=bounds, A_eq=[np.r_[np.ones(count), 0.]], b_eq=[1.],
        bounds=[(0., 1.)] * count + [(None, None)],
        method='highs',
    )
    if not result.success:
        raise RuntimeError(result.message)
    # Any nonnegative convex combination of the inequalities gives a lower bound.
    certificate = np.maximum(-result.ineqlin.marginals, 0.)
    certificate /= certificate.sum()
    support = np.flatnonzero(certificate)
    lower = float(np.min(certificate @ (rows - bounds[:, None])))
    if abs(lower - result.fun) > 1e-6:
        raise ValueError('Linear lower bound does not verify')
    coefficients = result.x[:count]
    return coefficients, lower, [
        {'row': int(index), 'weight': float(certificate[index]),
         'residual': (rows[index] - bounds[index]).tolist()}
        for index in support
    ]


def render_whole(frames, target, truth, zones):
    rows, bounds = inequality_system(frames, target, truth, zones)
    weights, lower, certificate = solve(rows, bounds)
    predictions = {
        light: {
            state: np.rint(np.tensordot(weights, images, axes=1)).astype(np.uint8)
            for state, images in states.items()
        }
        for light, states in frames.items()
    }
    return predictions, {'weights': weights.tolist(), 'lower_bound': lower,
                         'constraints': len(bounds), 'certificate': certificate}


def run(out, *, render=render_whole):
    prior = BASE / 'gros-film-b-cross-v2'
    new = BASE / 'gros-two-image-edit-regression-v1'
    baseline = checked_json(prior / 'results.json', BASELINE_SHA)
    regression = checked_json(new / 'results.json', REGRESSION_SHA)
    groups = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    if regression['parent'] != BASELINE_SHA:
        raise ValueError('Regression was not run on the same Film B scenes')
    out.mkdir()
    records = {}
    for case, record in baseline['cases'].items():
        person = case.split('-')[0]
        target = checked_rgb(prior / f'{person}-cold-off.png', baseline['inputs'][f'{person}-cold']['hashes']['off'])
        truth = checked_rgb(prior / f'{case}-expected.png', record['expected_sha'])
        zones = masks(groups[person]['cold'], groups['screen'], target.shape[:2])
        frames = {}
        for light in LIGHTS:
            frames[light] = {}
            for state in STATES:
                editions = []
                for method in CANDIDATES[:-1]:
                    frozen = method in ('stable', '13')
                    label = state if frozen or state == 'off' else 'film_b'
                    key = f'{method}-{light}-{label}'
                    path = (prior if frozen else new) / f'{case}-{key}.png'
                    ledger = record if frozen else regression['cases'][case]
                    editions.append(checked_rgb(path, ledger['hashes'][key]))
                frames[light][state] = np.stack([*editions, target])
        predictions, details = render(frames, target, truth, zones)
        output_hashes = {}
        for light, states in predictions.items():
            for state, image in states.items():
                name = f'{case}-{light}-{state}.png'
                output_hashes[name] = save(out / name, image)
        scores = score(target, truth, predictions, zones)
        if details.get('lower_bound', 0.) > 1e-6 and all(row['passed'] for row in scores.values()):
            raise ValueError('Rounded rendering contradicts the necessary constraints')
        for name, detail in details.get('regions', {}).items():
            if name in scores and detail['lower_bound'] > 1e-6 and scores[name]['passed']:
                raise ValueError(f'Rounded rendering contradicts the constraints in {name}')
        records[case] = details | {'scores': scores, 'hashes': output_hashes}
    report = {'sources': [BASELINE_SHA, REGRESSION_SHA, REGIONS_SHA],
              'candidate_order': CANDIDATES, 'records': records}
    (out / 'results.json').write_bytes((json.dumps(report, indent=2, allow_nan=False) + '\n').encode())


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
