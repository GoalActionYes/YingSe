"""Render actual authored targets from fixed native reference sample policies."""

import argparse
from pathlib import Path

import numpy as np

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST
from probe_tst_author_neighbor_replay import PARENT, PARENT_SHA
from probe_tst_paired_donor_edit import (
    BASE, RENDER_SHA, SPLIT, SPLIT_SHA, checked_rgb, donor_assets, family, finish_spatial,
    json_to, load, neutral, replay, save, score,
)


def grid_centers(length):
    return (2 * np.arange(256) + 1) * length // 512


def dense_indices(shape, train):
    h, w = shape[:2]
    y = (2 * np.arange(h) + 1) * 128 // h
    x = (2 * np.arange(w) + 1) * 128 // w
    positions = np.flatnonzero(train[y[:, None], x[None, :]])
    if len(positions) > 131072:
        positions = positions[np.arange(131072) * len(positions) // 131072]
    return positions


def sample_pairs(original, edited, train):
    if original.shape != edited.shape:
        raise ValueError('Different original and edit sizes')
    y, x = (grid_centers(length) for length in original.shape[:2])
    centered = [im[y[:, None], x[None, :]][train] for im in (original, edited)]
    positions = dense_indices(original.shape, train)
    dense = [im.reshape(-1, 3)[positions] for im in (original, edited)]
    return {'native_equal': centered, 'native_dense': dense}


def run(out):
    parent = load(PARENT / 'results.json', PARENT_SHA)
    rows = load(SPLIT / 'private-assignment.json', SPLIT_SHA)['development']
    entries = {row['original_path']: row for row in load(MANIFEST, HASHES[2])['files']}
    sources, read = donor_assets()
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    y, x = np.indices((256, 256))
    train = (x // 16 + y // 16) % 2 == 0
    out.mkdir()
    cases = {}
    for key, slot_info in parent['slots'].items():
        if 'scores' not in slot_info:
            continue
        index, slot = map(int, key.split('-'))
        row = rows[index]
        guide = row['refs'][slot]
        source, = sources[family(guide, 'reference')]
        donor, graded, target = [read(name)[0] for name in (source, guide, row['content'])]
        old = slot_info['scores']['donor']
        archived = checked_rgb(PARENT / f'{key}-donor.png', old['sha256'])
        np.testing.assert_array_equal(replay(donor, graded, target, train), archived)
        pair = []
        for name in (source, guide):
            item = entries[name]
            pair.append(checked_rgb(COHORT.parent / item['prepared_path'], item['sha256']))
        variants = sample_pairs(*pair, train)
        base = checked_rgb(archive / f'{key}.png', frozen['predictions'][f'{key}.png'])
        author = read(row['different_truth' if slot == 2 else 'same_truth'])[0]
        truth = finish_spatial(target, author, .8)
        region = neutral(target) & ~train
        if int(region.sum()) != slot_info['neutral_pixels']:
            raise ValueError('Changed neutral coverage')
        case = {'old': old}
        for name, (before, after) in variants.items():
            frame = replay(before, after, target, np.ones(len(before), dtype=bool))
            stats = score(frame, target, truth, base, ~train, region)
            case[name] = {**stats, 'extra_vs_old': stats['holdout_mae'] - old['holdout_mae'],
                          'samples': len(before), 'unique_colors': len(np.unique(before, axis=0)),
                          'sha256': save(out / f'{key}-{name}.png', frame)}
        cases[key] = case
    if len(cases) != 10:
        raise ValueError('Changed development cases')
    passed = all(case['native_dense']['passed'] and case['native_dense']['extra_vs_old'] <= .25
                 for case in cases.values())
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'cases': cases, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
