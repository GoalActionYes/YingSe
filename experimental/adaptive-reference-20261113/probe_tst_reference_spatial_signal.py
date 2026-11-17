"""Measure offline spatial information in authored differences between two grades."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST, load
from background_intervention import target_regions
from independent_coco import digest
from probe_inretouch_registration import gradient, shifts
from train_tst_reference_equivalence import SPLIT, SPLIT_SHA


def lookup_mae(keys, delta):
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    starts = np.r_[0, counts.cumsum()[:-1]]
    lower, upper = starts + (counts - 1) // 2, starts + counts // 2
    error = 0.
    for values in delta.T:
        sorted_values = values[np.lexsort((values, groups))]
        medians = (sorted_values[lower] + sorted_values[upper]) / 2
        error += np.abs(values - medians[groups]).sum()
    return float(error / delta.size), float(np.mean(counts[groups] > 1))


def measure(source, a, b, seed):
    h, w = source.shape[:2]
    mask = target_regions(source, np.zeros((h, w), bool))['neutral_proxy']
    rgb = source[mask].astype(np.uint32)
    if not len(rgb):
        return dict(neutral_pixels=0)
    keys = rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]
    delta = b[mask].astype(np.int16) - a[mask].astype(np.int16)
    rows, cols = np.nonzero(mask)
    quadrant = (rows >= h // 2).astype(np.uint32) * 2 + (cols >= w // 2)
    plain_order = np.argsort(keys, kind='stable')
    shuffled_order = np.lexsort((np.random.default_rng(seed).random(len(keys)), keys))
    shuffled = np.empty_like(quadrant)
    shuffled[plain_order] = quadrant[shuffled_order]
    base, repeated = lookup_mae(keys, delta)
    local, _ = lookup_mae(keys * 4 + quadrant, delta)
    null, _ = lookup_mae(keys * 4 + shuffled, delta)
    assert max(local, null) <= base + 1e-10
    return dict(neutral_pixels=len(rgb), repeated_pixel_fraction=repeated, global_mae=base,
                spatial_mae=local, shuffled_mae=null,
                mean_author_delta=float(np.abs(delta).mean()))


def structure_score(source, result):
    outputs = shifts(gradient(source), gradient(result))
    whole = outputs[0]
    passed = whole['zero'] >= .90 and (whole['dy'], whole['dx']) == (0, 0)
    passed = passed and all(row['peak_gain'] <= .01 for row in outputs[1:])
    return dict(passed=passed, whole=whole, maximum_quadrant_shift_gain=max(r['peak_gain'] for r in outputs[1:]))


def run(out):
    rows = load(SPLIT / 'private-assignment.json', SPLIT_SHA)['development']
    entries = {row['original_path']: row for row in load(MANIFEST, HASHES[2])['files']}
    assert len(rows) == 24
    start = time.monotonic()
    cases = []
    for index, row in enumerate(rows):
        originals, reduced = [], []
        for name in ('content', 'same_truth', 'different_truth'):
            entry = entries[row[name]]
            path = COHORT.parent / entry['prepared_path']
            assert digest(path) == entry['sha256']
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                originals.append(np.array(rgb))
                reduced.append(np.array(rgb.resize((256, 256), Image.Resampling.BILINEAR)))
        assert len({value.shape for value in originals}) == 1
        geometry = [structure_score(reduced[0], value) for value in reduced[1:]]
        native = measure(*originals, seed=20261118 + index)
        small = measure(*reduced, seed=20261118 + index)
        passes = (all(r['passed'] for r in geometry) and native.get('global_mae', 0) >= 2.
                  and all(native[name] - native['spatial_mae'] >= .25
                          and native['spatial_mae'] <= .95 * native[name]
                          for name in ('global_mae', 'shuffled_mae')))
        cases.append(dict(case=index, size=list(originals[0].shape[:2]), geometry=geometry,
                          native=native, bilinear_256=small, passed=passes))
        assert time.monotonic()-start <= 180
    count = sum(row['passed'] for row in cases)
    report = dict(cases=cases, qualifying_cases=count, proceed=count >= 6, assignment_sha256=SPLIT_SHA,
                  script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('tst-reference-spatial-signal-protocol.md')),
                  seconds=time.monotonic()-start)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(count, report['seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
