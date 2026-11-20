"""Evaluate the editor-response envelope after giving the correct capture intent."""
import argparse
from pathlib import Path

import numpy as np

from probe_beyond_rgb_matched_cast import compare
from probe_mid_matched_wall_cast import PARENT as LEDGER, targets
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save

PARENT = '9f1ca17f3b643017ec42f0a0003184ac3da455e5f93c40f93fe9c02669a0811a'


def clipped(rgb):
    return np.clip(rgb, 0, 255)


def solution(left, right, want):
    step = right - left
    # L1 of clipping can change slope at RGB endpoints as well as at the answer.
    fractions = [np.zeros(left.shape[:2]), np.ones(left.shape[:2])]
    for threshold in (np.zeros_like(want), np.full_like(want, 255), want):
        points = np.divide(threshold - left, step, where=step != 0, out=np.zeros_like(left))
        fractions.extend(np.moveaxis(np.clip(points, 0, 1), -1, 0))
    best = clipped(left)
    best_error = np.abs(best - want).sum(axis=-1)
    weight = np.zeros(left.shape[:2])
    for fraction in fractions[1:]:
        rgb = clipped(left + fraction[..., None] * step)
        cost = np.abs(rgb - want).sum(axis=-1)
        improved = cost < best_error
        best[improved] = rgb[improved]
        best_error[improved] = cost[improved]
        weight[improved] = fraction[improved]
    return best, weight


def run(out):
    parent = BASE / 'mid-matched-wall-cast-v1'
    prior = checked_json(parent / 'results.json', PARENT)
    scenes = checked_json(BASE / 'mid-gray-anchor-v1.json', LEDGER)['scenes']
    out.mkdir()
    cases = {}
    for name, (image, areas) in targets(scenes).items():
        reference = prior['cases'][name]
        expected = checked_rgb(parent / f'{name}-expected.png', reference['expected_sha'])
        start = image.astype(np.float64)
        values = {}
        for model, path in reference['models'].items():
            states = {label: checked_rgb(parent / f'{name}-{model}-{label}.png', path['hashes'][label])
                      for label in ('base', 'edited')}
            values[model] = start + states['edited'].astype(np.float64) - states['base']
        best, weight = solution(values['stable'], values['13'], expected)
        arms = {key: np.rint(clipped(rgb)).astype(np.uint8) for key, rgb in values.items()}
        arms['offline_best'] = np.rint(best).astype(np.uint8)
        regions = {}
        for label, mask in areas.items():
            continuous = float(np.abs(best[mask] - expected[mask]).mean())
            measured = {key: compare(image, expected, image, image, rgb, mask)['style']
                        for key, rgb in arms.items()}
            if any(continuous - .5 > row['response_mae'] + 1e-12 for row in measured.values()):
                raise ValueError('Quantization exceeds analytic bound')
            regions[label] = dict(continuous_min=continuous, rounded_lower_bound=max(continuous - .5, 0),
                                  fractions=np.quantile(weight[mask], [.25, .50, .75]).tolist(),
                                  measured=measured)
        cases[name] = dict(regions=regions,
                           hashes={key: save(out / f'{name}-{key}.png', rgb) for key, rgb in arms.items()},
                           panel=panel(out / f'{name}-comparison.png', {
                               'target': {'original': image, 'desired edit': expected}, 'responses': arms}),
                           passed=all(row['measured']['offline_best']['passed'] for row in regions.values()))
    json_to(out / 'results.json', dict(parent=PARENT, script_sha=digest(Path(__file__)),
            protocol_sha=digest(Path(__file__).with_name('mid-edit-response-envelope-protocol.md')), cases=cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
