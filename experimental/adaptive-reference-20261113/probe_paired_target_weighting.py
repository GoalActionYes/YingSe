"""Evaluate target RGB occupancy weighting on frozen reference before/after pairs."""

import argparse
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_aligned_cubic_pilot import POWERS, RIDGE, cubic_mapping, design, learn, thumbnail_pair
from probe_bg_real_paired_two_look import SWITCH_SHA
from probe_paired_cubic_cross_content import apply
from probe_paired_operator_support import PARENT, readings
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, panel, save
from verify_development_inputs import manifest


def bins(rgb):
    r, g, b = (rgb.reshape(-1, 3).astype(np.int32) // 32).T
    return (r * 8 + g) * 8 + b


def occupancy_weights(reference, target, train):
    reference_bins = bins(reference)
    counts = np.bincount(reference_bins[train], minlength=512)
    target_counts = np.bincount(bins(target), minlength=512)
    smooth = 1 / 512
    ratio = (target_counts / target_counts.sum() + smooth) / (counts / counts.sum() + smooth)
    return np.clip(ratio, .25, 4)[reference_bins], counts


def weighted_learn(original, edited, train, weight):
    a, b = (rgb.reshape(-1, 3).astype(np.float64) / 255 for rgb in (original, edited))
    x = design(a)
    nonlinear = np.diag([sum(powers) >= 2 for powers in POWERS]).astype(float)
    coefficients = []
    for channel in range(3):
        usable = train & (b[:, channel] > 2 / 255) & (b[:, channel] < 253 / 255)
        local_weight = weight[usable]
        root = np.sqrt(local_weight)
        u = x[usable] * root[:, None]
        delta = (b[:, channel] - a[:, channel])[usable] * root
        regularizer = np.sqrt(RIDGE * local_weight.sum()) * nonlinear
        coefficient, _, rank, _ = np.linalg.lstsq(
            np.vstack((u, regularizer)), np.r_[delta, np.zeros(len(POWERS))], rcond=None
        )
        if rank != len(POWERS) or not np.isfinite(coefficient).all():
            raise ValueError("Unstable weighted fit")
        coefficients.append(coefficient)
    return np.stack(coefficients, axis=1)


def render(original, edited, target, archived, _style):
    if original.shape != edited.shape:
        raise ValueError("Verified reference pair has different sizes")
    a, b = thumbnail_pair(original, edited)
    y, x = np.indices(a.shape[:2])
    train = ((x // 8 + y // 8) % 2 == 0).ravel()
    control = learn(a, b, train)
    np.testing.assert_array_equal(weighted_learn(a, b, train, np.ones(train.size)), control)
    np.testing.assert_array_equal(apply(target, cubic_mapping(control)), archived)
    weight, counts = occupancy_weights(a, target, train)
    frame = apply(target, cubic_mapping(weighted_learn(a, b, train, weight)))
    return frame, counts


def passing(row):
    score = row['score']
    return .75 <= score['projection'] <= 1.25 and score['response_mae'] <= 3


def measure(out, key, row, archive, history, old_pair, renderer, method):
    source_dir = BASE / 'bg-real-editor-look' / key
    switch_dir = BASE / 'bg-real-two-look-switch-v3' / key
    film_dir = BASE / 'bg-real-film-b-qualification-v1'
    prior = archive['results']['13'][key]['asset_hashes']
    qualified = history['qualification'][key]
    target = checked_rgb(source_dir / 'target.png', row['artifacts']['target.png'])
    known_a = checked_rgb(switch_dir / 'expected-a.png', prior['expected-a'])
    known_b = checked_rgb(film_dir / f'{key}-expected.png', qualified['expected_sha256'])
    y, x = np.indices(target.shape[:2])
    holdout = (x // 16 + y // 16) % 2 == 1
    whole = np.ones_like(holdout)
    neutral = np.logical_and.reduce([target_regions(v, ~whole)['neutral_proxy'] for v in (target, known_a, known_b)])
    areas = {'whole': whole, 'neutral': neutral}
    for name, rect in regions()[key].items():
        mask = np.zeros_like(holdout)
        mask[rect] = True
        areas[name] = mask

    out.mkdir()
    results, board = {}, {}
    for role in ('C1', 'F1'):
        name = f'{role}-off-reference.png'
        original = checked_rgb(source_dir / name, row['artifacts'][name])
        reference_a = checked_rgb(switch_dir / f'{role}-a-reference.png', prior[role])
        reference_b = checked_rgb(film_dir / f'{key}-{role}-reference.png',
                                  qualified['reference_diagnostics'][role]['edited_sha256'])
        old_a = checked_rgb(BASE / 'bg-real-paired-two-look-v1' / key / f'{role}-a.png',
                            old_pair['cases'][key]['roles'][role]['frame_a_sha256'])
        old_b = checked_rgb(film_dir / f'paired-{key}-{role}-on.png',
                            history['comparison']['paired']['cases'][key]['outputs'][f'{role}-on'])
        new_a, counts_a = renderer(original, reference_a, target, old_a, 'film_a')
        new_b, counts_b = renderer(original, reference_b, target, old_b, 'film_b')
        np.testing.assert_array_equal(counts_a, counts_b)
        old = readings(known_a, known_b, old_a, old_b, areas, holdout)
        new = readings(known_a, known_b, new_a, new_b, areas, holdout)
        white_passes = None
        regressions, support = [], {}
        for label, before in old.items():
            local_bins = bins(target[areas[label] & holdout])
            support[label] = int(np.count_nonzero(counts_a[local_bins]))
            after = new[label]
            if not before.get('eligible'):
                continue
            if label == '白球服' and key == '872-1268':
                white_passes = passing(after)
                continue
            worse = after['score']['response_mae'] - before['score']['response_mae']
            if worse > .25 or (passing(before) and not passing(after)):
                regressions.append(label)
        hashes = {style: save(out / f'{role}-{style}.png', rgb) for style, rgb in (('a', new_a), ('b', new_b))}
        results[role] = {
            'old': old, method: new, 'white_passes': white_passes, 'regressions': regressions,
            'represented_pixels': support, 'hashes': hashes,
        }
        board[f'{role} old'] = {'target': target, 'A': old_a, 'B': old_b, 'known A': known_a, 'known B': known_b}
        board[f'{role} {method}'] = {'target': target, 'A': new_a, 'B': new_b, 'known A': known_a, 'known B': known_b}
    return dict(roles=results, panel_sha256=panel(out / 'comparison.png', board))


def run(out, renderer=render, method='weighted'):
    archive = checked_json(BASE / 'bg-real-two-look-switch-v3/results.json', SWITCH_SHA)
    history = checked_json(BASE / 'bg-real-film-b-qualification-v1/results.json', archive['film_b_report_sha256'])
    old_pair = checked_json(BASE / 'bg-real-paired-two-look-v1/results.json', PARENT)
    out.mkdir()
    cases = {key: measure(out / key, key, row, archive, history, old_pair, renderer, method)
             for key, row in manifest(BASE, 'bg-real-editor-look')['cases'].items()}
    passed = (all(role['white_passes'] is True for role in cases['872-1268']['roles'].values())
              and all(not role['regressions'] for case in cases.values() for role in case['roles'].values()))
    json_to(out / 'results.json', dict(cases=cases, passed=passed, parent_sha256=PARENT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
