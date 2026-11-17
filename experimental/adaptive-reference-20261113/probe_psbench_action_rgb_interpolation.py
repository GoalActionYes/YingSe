"""Test a fixed RGB interpolation of the observed reference edit, offline only."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_psbench_action_rgb_transfer import color_keys, edit_lookup, score
from probe_psbench_teal_portrait import verify_source
from reference_color_app.media import decode_image


STEP = 16
GRID_SIZE = 17


def make_grid(keys, edits):
    codes, values = edit_lookup(keys, edits)
    rgb = ((codes[:, None] >> np.array([16, 8, 0])) & 255)
    xyz = (rgb + STEP // 2) // STEP
    indices = np.ravel_multi_index(xyz.T, (GRID_SIZE,) * 3)
    counts = np.bincount(indices, minlength=GRID_SIZE ** 3)
    sums = np.stack([np.bincount(indices, weights=channel, minlength=counts.size)
                     for channel in values.T], axis=-1)
    return (sums / np.maximum(counts[:, None], 1)).reshape((GRID_SIZE,) * 3 + (3,)), (
        counts >= 4
    ).reshape((GRID_SIZE,) * 3)


def interpolate(rgb, grid, known):
    coordinate = rgb.astype(np.float64) / STEP
    low = coordinate.astype(np.int32)
    fraction = coordinate - low
    weighted = np.zeros_like(coordinate)
    coverage = np.zeros(len(rgb), dtype=np.float64)
    for corner in product((0, 1), repeat=3):
        indices = tuple((low + corner).T)
        weights = np.prod(np.where(corner, fraction, 1 - fraction), axis=-1)
        weights *= known[indices]
        coverage += weights
        weighted += grid[indices] * weights[:, None]
    valid = coverage >= .5
    weighted[valid] /= coverage[valid, None]
    return weighted, valid


def run(folder, out, baseline_evidence, exact_evidence):
    prior = json.loads(baseline_evidence.read_text(encoding='utf-8'))
    exact = json.loads(exact_evidence.read_text(encoding='utf-8'))
    if verify_source(folder, '105') != prior['files'] or prior['strength'] != .8:
        raise ValueError('Unexpected source files or blend strength')
    if digest(baseline_evidence) != exact['baseline_sha256']:
        raise ValueError('Exact support and baseline have different sources')
    source, edited = (decode_image(folder / f'105_{role}.jpg') for role in ('start', 'end'))
    if source.shape != edited.shape or source.shape != (1080, 1440, 3):
        raise ValueError('Unexpected collage dimensions')
    regions = {}
    for name, (left, right) in prior['crop_columns'].items():
        original = source[:, left:right]
        baseline = folder / 'action-pair-v1' / f'{name}.png'
        support_file = folder / 'action-rgb-v1' / f'{name}-support.png'
        if digest(baseline) != prior['predictions'][name]:
            raise ValueError(f'{name}: 13 baseline changed')
        if digest(support_file) != exact['artifacts'][support_file.name]:
            raise ValueError(f'{name}: exact coverage changed')
        with Image.open(support_file) as im:
            exact_support = np.asarray(im) == 255
        delta = np.rint((edited[:, left:right].astype(np.int16) - original) * .8).astype(np.int16)
        regions[name] = dict(original=original, expected=original + delta, delta=delta,
                             keys=color_keys(original), exact=exact_support, supported=np.zeros_like(exact_support),
                             variants=dict(original=original, current13=decode_image(baseline),
                                           cross_scene=original.copy(), same_scene=original.copy()))

    rows, cols = np.ogrid[:1080, :704]
    folds = ((rows // 32 + cols // 32) % 2).astype(bool)
    for test in (folds, ~folds):
        grids = {name: make_grid(region['keys'][~test], region['delta'][~test])
                 for name, region in regions.items()}
        for name, reference in (('portrait', 'tower'), ('tower', 'portrait')):
            region = regions[name]
            original = region['original'][test]
            cross, cross_valid = interpolate(original, *grids[reference])
            same, same_valid = interpolate(original, *grids[name])
            supported = cross_valid & same_valid
            region['supported'][test] = supported
            for method, values in (('cross_scene', cross), ('same_scene', same)):
                output = original.copy()
                output[supported] = np.rint(original[supported] + values[supported]).clip(0, 255)
                region['variants'][method][test] = output

    out.mkdir(exist_ok=False)
    artifacts, scores = {}, {}
    for name, region in regions.items():
        supported = region['supported']
        added = supported & ~region['exact']
        variants = region['variants']
        for role, array in [('cross_scene', variants['cross_scene']), ('same_scene', variants['same_scene']),
                            ('supported', supported.astype(np.uint8) * 255),
                            ('added', added.astype(np.uint8) * 255)]:
            path = out / f'{name}-{role}.png'
            Image.fromarray(array).save(path)
            artifacts[path.name] = digest(path)
        scores[name] = {
            label: score(region['original'], region['expected'], variants, mask)
            for label, mask in [('all', np.ones_like(supported)), ('supported', supported), ('added', added)]
        }
    improved = True
    for case in scores.values():
        improved &= case['supported']['neutral_proxy']['coverage'] >= .25
        for row in (case['all']['whole'], case['all']['neutral_proxy'], case['added']['neutral_proxy']):
            methods = row['methods']
            if not methods:
                improved = False
                continue
            cross = methods['cross_scene']
            improved &= cross['mae'] <= min(methods['original']['mae'], methods['current13']['mae']) - 1
            improved &= (cross['projection'] or 0) > 0
    summary = dict(scores=scores, artifacts=artifacts, passed=improved,
                   baseline_sha256=digest(baseline_evidence), exact_sha256=digest(exact_evidence),
                   script_sha256=digest(__file__),
                   protocol_sha256=digest(Path(__file__).with_name('psbench-action-rgb-interpolation-protocol.md')))
    (out / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: {region: dict(
        coverage=row['coverage'], cross_mae=row['methods'].get('cross_scene', {}).get('mae')
    ) for region, row in case['all'].items()} for name, case in scores.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('baseline_evidence', type=Path)
    parser.add_argument('exact_evidence', type=Path)
    run(**vars(parser.parse_args()))
