"""Test an existing paired color operator on the qualified two-look cases."""
import argparse
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_bg_real_two_look_switch import FILM_B_REPORT_SHA
from probe_paired_cubic_cross_content import apply, paired_fit
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, panel, save
from verify_development_inputs import manifest

SWITCH_SHA = 'e1ee9194d2117910264631804536485be059b204c6ce94d31e0f2fdbab26fd81'


def measure(out, key, target, person, source, reference_assets, scores, history):
    b_root = BASE / 'bg-real-film-b-qualification-v1'
    a_root = BASE / 'bg-real-two-look-switch-v3' / key
    a = checked_rgb(a_root / 'expected-a.png', scores['asset_hashes']['expected-a'])
    b = checked_rgb(b_root / f'{key}-expected.png', history['qualification'][key]['expected_sha256'])
    neutral = np.logical_and.reduce([target_regions(frame, person)['neutral_proxy'] for frame in (target, a, b)])
    if neutral.sum() < 256:
        raise ValueError('Lost fixed common neutral support')
    areas = {'whole': np.s_[:, :], **regions()[key]}
    paired = history['comparison']['paired']['cases'][key]
    records, board = {}, {}
    out.mkdir()
    for role in ('C1', 'F1'):
        name = f'{role}-off-reference.png'
        off = checked_rgb(source / key / name, reference_assets[name])
        ref_a = checked_rgb(a_root / f'{role}-a-reference.png', scores['asset_hashes'][role])
        ref_b = checked_rgb(b_root / f'{key}-{role}-reference.png',
                            history['qualification'][key]['reference_diagnostics'][role]['edited_sha256'])
        mix = {}
        for style, reference in (('a', ref_a), ('b', ref_b)):
            mapping, fit = paired_fit(off, reference)
            frame = apply(target, mapping)
            if style == 'b':
                b_stored = checked_rgb(b_root / f'paired-{key}-{role}-on.png', paired['outputs'][f'{role}-on'])
                np.testing.assert_array_equal(frame, b_stored)
            mix[style] = (frame, fit)
        frames = {style: record[0] for style, record in mix.items()}
        readings = {}
        for label, where in areas.items():
            aa, bb, left, right = [image[where] for image in (a,b,frames['a'],frames['b'])]
            readings[label] = metrics(aa, bb, left, right, np.ones(aa.shape[:2], bool))
        readings['neutral'] = metrics(a, b, frames['a'], frames['b'], neutral)
        records[role] = dict(switch=readings, fit={style: record[1] for style, record in mix.items()},
                             frame_a_sha256=save(out / f'{role}-a.png', frames['a']))
        board[role] = {'A paired': frames['a'], 'B paired': frames['b'], 'known A': a, 'known B': b}
    samples = {role: {'off':target, 'on': row['A paired']} for role, row in board.items()}
    a_result = evaluate(target, {'off':target,'on':a}, samples, person)
    failures = [{ 'role': role, 'area': label } for role, entry in records.items()
                for label, values in entry['switch'].items()
                if label not in ('whole','neutral') and values['original_matches_authored_mae'] >= 1 and not values['passed']]
    switches = all(row['switch'][label]['passed'] for row in records.values() for label in ('whole','neutral'))
    return dict(roles=records, original_a_gate=a_result, original_b_gate=paired['scores'],
                large_region_swap_pass=switches, local_failures=failures,
                passed=switches and a_result['passed'] and paired['scores']['passed'] and not failures,
                comparison_sha256=panel(out / 'comparison.png', board))


def run(out):
    source = BASE / 'bg-real-editor-look'
    prior = checked_json(BASE / 'bg-real-two-look-switch-v3/results.json', SWITCH_SHA)
    film = checked_json(BASE / 'bg-real-film-b-qualification-v1/results.json', FILM_B_REPORT_SHA)
    registered = {row['pair']: row for row in checked_json(PARENT / 'results.json', PARENT_SHA)['cases']}
    out.mkdir()
    cases = {}
    for key, row in manifest(BASE, 'bg-real-editor-look')['cases'].items():
        target = checked_rgb(source / key / 'target.png', row['artifacts']['target.png'])
        person = mask_for(registered[key], key.split('-')[0])
        cases[key] = measure(out / key, key, target, person, source, row['artifacts'], prior['results']['13'][key], film)
    count = sum(row['passed'] for row in cases.values())
    json_to(out / 'results.json', dict(parent_sha256=SWITCH_SHA, cases=cases, joint=count, passed=count>=3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
