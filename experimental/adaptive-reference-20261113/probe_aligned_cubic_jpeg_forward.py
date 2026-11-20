"""Isolate the effect of simulating JPEG in the aligned-fit validator."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_aligned_cubic_capture_stress import Stress
from probe_aligned_cubic_fine_residual import fine_residual, fixtures, recode
from probe_aligned_cubic_jpeg_scale import ENCODINGS, PREVIOUS, final_scores, half_mean
from probe_aligned_cubic_pilot import FIT_FAILURES, cubic_mapping, learn, thumbnail_pair
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.aligned_reference import fit_aligned
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial


ARCHIVE = 'daa25c2d6642afdde59347b03504897f6d6bb89a5adac1793f3c1c26694454f9'
ARMS = ('legacy_thumbnail', 'native_then_resize', 'jpeg_then_resize')


def errors(proposed, wanted, train):
    delta = np.abs(proposed.astype(np.int16) - wanted)
    held = delta[~train]
    mae = float(held.mean())
    p95 = float(np.quantile(held, .95))
    h, w = train.shape
    worst = max(float(delta[j:j + 8, i:i + 8].mean()) for j in range(0, h, 8) for i in range(0, w, 8))
    return dict(mae=mae, p95=p95, worst_tile=worst, passed=mae <= 1.5 and p95 <= 4 and worst <= 6)


def compare(target, reference):
    if np.array_equal(target, reference):
        return 'identity', None, {}
    affine = fit_aligned(target, reference)
    if affine['accepted'] or affine['reason'] not in FIT_FAILURES:
        return 'affine' if affine['accepted'] else 'original_fallback', None, {}
    a, b = thumbnail_pair(target, reference)
    y, x = np.indices(a.shape[:2])
    train = (x // 8 + y // 8) % 2 == 0
    weights = learn(a, b, train.ravel())
    if weights is None:
        return 'original_fallback', None, {}
    table = cubic_mapping(weights)
    native = apply_mapping(target, table, 1)
    images = (apply_mapping(a, table, 1), thumbnail_pair(native, reference)[0],
              thumbnail_pair(recode(native), reference)[0])
    checks = {name: errors(image, b, train) for name, image in zip(ARMS, images, strict=True)}
    original = 'cubic' if checks['legacy_thumbnail']['passed'] else 'original_fallback'
    return original, native, checks


def examine(target, reference, expected_reference, prior, engine, folder, name):
    route, native, checks = compare(target, reference)
    if route != prior['before']:
        raise ValueError(f'Original fit decision changed: {name}')
    row = dict(before=route, fit=checks)
    if native is None or not any(reading['passed'] for reading in checks.values()):
        return row
    evidence = fine_residual(half_mean(native), half_mean(reference))
    baseline = engine.baseline(target, reference)
    candidate = finish_spatial(target, native, STRENGTH)
    output = candidate if checks['jpeg_then_resize']['passed'] and evidence['passed'] else baseline
    tiles = {'target': target, 'JPEG reference': reference, '13': baseline, 'candidate': output}
    row['spatial'] = evidence
    row['jpeg_matches_13'] = bool(np.array_equal(output, baseline))
    if expected_reference is not None:
        expected = finish_spatial(target, expected_reference, STRENGTH)
        tiles['known edit'] = expected
        row['cubic_scores'] = final_scores(target, expected, candidate)
        row['actual_scores'] = final_scores(target, expected, output)
    row['output_sha256'] = save(folder / f'{name}.png', output)
    row['panel_sha256'] = panel(folder / f'{name}-comparison.png', {'fixed JPEG setup': tiles})
    return row


def count(rows, fixed):
    positives, negatives = rows['positives'], rows['negatives']
    counts = {}
    for arm in ARMS:
        passed = [name for name in sorted(fixed) if positives[name]['fit'].get(arm, {}).get('passed')]
        final = [name for name in passed if positives[name]['spatial']['passed']
                 and all(value['passed'] for value in positives[name]['cubic_scores'].values())]
        accepted = [name for name, row in negatives.items() if row['fit'].get(arm, {}).get('passed')]
        false = [name for name in accepted if negatives[name]['spatial']['passed']]
        counts[arm] = dict(fit_positive=passed, final_positive=final, fit_negative=accepted, false_positive=false,
                           passed=len(final) >= 6 and not false)
    return counts


@torch.inference_mode()
def run(out):
    original = BASE / 'aligned-cubic-jpeg-scale-v1/results.json'
    previous = checked_json(original, ARCHIVE)
    lossless = checked_json(BASE / 'aligned-cubic-fine-residual-v1/results.json', PREVIOUS)
    fixed = {name for name, row in lossless['positives'].items() if row['route_after'] == 'cubic'}
    if len(fixed) != 8:
        raise ValueError('Changed fixed positive set')
    out.mkdir()
    torch.set_num_threads(4)
    engine = Stress(out)
    folders = {key: out / key for key in ENCODINGS}
    for folder in folders.values():
        folder.mkdir()
    cases = {condition: {'positives': {}, 'negatives': {}} for condition in ENCODINGS}
    for group, name, target, reference in fixtures():
        jpeg_target, jpeg_reference = recode(target), recode(reference)
        expected = reference if group == 'positives' else None
        for condition, base in zip(ENCODINGS, (target, jpeg_target), strict=True):
            prior = previous['cases'][condition][group][name]
            cases[condition][group][name] = examine(
                base, jpeg_reference, expected, prior, engine, folders[condition], name)
    counts = {}
    for condition, rows in cases.items():
        if {key: len(items) for key, items in rows.items()} != {'positives': 14, 'negatives': 47}:
            raise ValueError('Changed registered pairs')
        counts[condition] = count(rows, fixed)
        original = previous['counts'][condition]
        legacy = counts[condition]['legacy_thumbnail']
        if legacy['fit_positive'] != original['available_original_positives']:
            raise ValueError('Changed fit eligibility')
        if legacy['final_positive'] != original['arms']['pooled']['passing_positive']:
            raise ValueError('Changed coarse residual baseline')
    passed = all(row['jpeg_then_resize']['passed'] for row in counts.values())
    json_to(out / 'results.json', dict(source_sha256=ARCHIVE, script_sha256=digest(Path(__file__)),
                                      counts=counts, cases=cases, passed=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
