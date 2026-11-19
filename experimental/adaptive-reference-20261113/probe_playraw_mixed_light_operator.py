"""Compare constant and RGB-conditioned replay of a real masked color edit."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_playraw_mixed_light_transfer import LUMA, PARENT, PARENT_SHA, STRENGTH, chroma, read, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, json_to, save
from spatial_render import finish_spatial


def features(rgb):
    colors = rgb.astype(np.float64) / 127.5 - 1
    return np.column_stack((np.ones(len(colors)), colors))


def fit(before, change, weight, train):
    x = features(before[train])
    y = change[train] / weight[train, None]
    penalty = .01 * len(x) * np.diag([0, 1, 1, 1])
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return np.median(y, axis=0), coefficients


def run(out):
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    before, _ = read(PARENT / 'off.tif', False)
    after, mask = read(PARENT / 'on.tif', True)
    if ([sha256(x.tobytes()).hexdigest() for x in (before, after)] != parent['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != parent['mask_pixels_sha256']):
        raise ValueError('Changed photo pair')
    light = before @ LUMA
    expected = chroma(finish_spatial(before, after, STRENGTH)) - chroma(before)
    y, x = np.indices(mask.shape)
    fold = (x // 8 + y // 8) % 2
    weight = mask.astype(np.float64) / 255
    active = mask > 0
    high = mask >= 204
    known = chroma(before[active])
    design = features(before[active])
    out.mkdir()
    cases = {}
    for train_half in (0, 1):
        train = high & (fold == train_half)
        other = fold != train_half
        zones = {'high': high & other, 'transition': active & ~high & other}
        fixed, affine = fit(before, expected, weight, train)
        results = {}
        for name, correction in (('constant', fixed), ('conditional', design @ affine)):
            result = before.copy()
            result[active], clipped = render(known + correction * weight[active, None], light[active])
            np.testing.assert_array_equal(result[~active], before[~active])
            actual = chroma(result) - chroma(before)
            results[name] = {'scores': {label: stats(actual, expected, area) for label, area in zones.items()},
                             'clipped_fraction': clipped, 'sha256': save(out / f'{train_half}-{name}.png', result)}
        constant, conditional = (results[name]['scores'] for name in ('constant', 'conditional'))
        p, error = [conditional['high'][key] for key in ('projection', 'response_mae')]
        passed = (.75 <= p <= 1.25 and error <= 3 and constant['high']['response_mae'] - error >= .25
                  and conditional['transition']['response_mae'] - constant['transition']['response_mae'] <= .25)
        cases[str(train_half)] = {'arms': results, 'training_pixels': int(train.sum()),
                                  'constant': fixed.tolist(), 'conditional': affine.tolist(), 'passed': passed}
    json_to(out / 'results.json', {'cases': cases, 'parent_sha256': PARENT_SHA,
                                   'passed': all(r['passed'] for r in cases.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
