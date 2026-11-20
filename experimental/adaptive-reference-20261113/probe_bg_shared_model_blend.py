"""Bound a shared RGB8 mixture of frozen grades and render its midpoint."""
import argparse
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_bg_style_direction_overlap import A_SHA, B_SHA
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, panel, save
from verify_development_inputs import manifest


ARMS = ('13', 'expanded')
ROLES = ('C1', 'F1')
STATES = ('off', 'a', 'b')


def endpoints(key, role, arm, a_root, b_root, aa, bb):
    row = bb['comparison'][arm]['cases'][key]['outputs']
    return {
        'off': checked_rgb(b_root / f'{arm}-{key}-{role}-off.png', row[f'{role}-off']),
        'a': checked_rgb(a_root / key / f'{arm}-{role}-a.png', aa['results'][arm][key]['frame_hashes'][role]),
        'b': checked_rgb(b_root / f'{arm}-{key}-{role}-on.png', row[f'{role}-on']),
    }


def convex_projection_bound(a, b, where, endpoint_scores):
    signal = b[where].astype(np.float64) - a[where]
    energy = np.square(signal).sum()
    if not energy:
        return None
    # Rounding A and B separately can change their difference by at most one code value.
    rounding = float(np.abs(signal).sum() / energy)
    projections = [score['projection'] for score in endpoint_scores]
    continuous_upper = max(projections)
    return {'continuous_upper': continuous_upper, 'rounding_allowance': rounding,
            'rgb8_upper': continuous_upper + rounding, 'floor_impossible': continuous_upper + rounding < .75}


def score_area(a, b, where, values, midpoint, prior):
    if isinstance(where, tuple):
        a, b = a[where], b[where]
        values = {arm: {state: img[where] for state, img in frames.items()} for arm, frames in values.items()}
        midpoint = {state: img[where] for state, img in midpoint.items()}
        where = np.ones(a.shape[:2], dtype=bool)
    scores = {arm: metrics(a, b, frames['a'], frames['b'], where) for arm, frames in values.items()}
    assert all(scores[arm] == prior[arm] for arm in ARMS), (scores, prior)
    actual = metrics(a, b, midpoint['a'], midpoint['b'], where)
    bound = convex_projection_bound(a, b, where, scores.values())
    if bound is not None and actual['projection'] > bound['rgb8_upper'] + 1e-12:
        raise ValueError('Rendered midpoint outside the projection bound')
    return {'endpoints': {arm: score['projection'] for arm, score in scores.items()},
            'bound': bound, 'midpoint': actual}


def midpoint_frame(first, second):
    return np.rint((first.astype(np.uint16) + second) * .5).astype(np.uint8)


def run(root, out):
    a_root, b_root = (root / name for name in ('bg-real-two-look-switch-v3', 'bg-real-film-b-qualification-v1'))
    aa, bb = checked_json(a_root / 'results.json', A_SHA), checked_json(b_root / 'results.json', B_SHA)
    masks = {row['pair']: row for row in checked_json(PARENT / 'results.json', PARENT_SHA)['cases']}
    boxes = regions()
    rows = manifest(root, 'bg-real-editor-look')['cases']
    assert set(rows) == set(aa['results']['13']) == set(boxes)
    out.mkdir()
    report = {}
    for key, archived in rows.items():
        local = out / key
        local.mkdir()
        prior = {arm: aa['results'][arm][key] for arm in ARMS}
        assert all(row['qualified'] for row in prior.values())
        target = checked_rgb(root / 'bg-real-editor-look' / key / 'target.png', archived['artifacts']['target.png'])
        a = checked_rgb(a_root / key / 'expected-a.png', prior['13']['asset_hashes']['expected-a'])
        b = checked_rgb(b_root / f'{key}-expected.png', bb['qualification'][key]['expected_sha256'])
        person = mask_for(masks[key], key.split('-')[0])
        neutral = np.logical_and.reduce([target_regions(img, person)['neutral_proxy'] for img in (target, a, b)])
        assert neutral.sum() >= 256
        areas = {'whole': np.s_[:, :], 'neutral': neutral, **boxes[key]}
        midpoint, scores, identities, pictures = {}, {}, {}, {}
        for role in ROLES:
            values = {arm: endpoints(key, role, arm, a_root, b_root, aa, bb) for arm in ARMS}
            blended = {state: midpoint_frame(*(values[arm][state] for arm in ARMS)) for state in STATES}
            midpoint[role] = blended
            identities[role] = {state: save(local / f'{role}-{state}.png', image) for state, image in blended.items()}
            scores[role] = {
                name: score_area(a, b, where, values, blended,
                                 {arm: prior[arm]['look_switch'][role][name] for arm in ARMS})
                for name, where in areas.items()
            }
            pictures[role] = {'known A': a, 'known B': b, 'half A': blended['a'],
                              'half B': blended['b'], 'half off': blended['off']}
        required = [scores[role][name] for role in ROLES for name in ('whole', 'neutral')]
        excluded = any(row['bound'] is not None and row['bound']['floor_impossible'] for row in required)
        swap = {state: evaluate(target, {'off': target, 'on': expected},
                                {role: {'off': frames['off'], 'on': frames[state]}
                                 for role, frames in midpoint.items()}, person)
                for state, expected in (('a', a), ('b', b))}
        passed = all(row['midpoint']['passed'] for row in required) and all(row['passed'] for row in swap.values())
        report[key] = {'excluded_by_projection': excluded, 'midpoint_passed': passed, 'scores': scores,
                       'midpoint_swap': swap, 'sha256': identities, 'board_sha256': panel(local / 'comparison.png', pictures)}
    json_to(out / 'results.json', {'parents': [A_SHA, B_SHA], 'source_sha256': digest(Path(__file__)),
                                   'protocol_sha256': digest(Path(__file__).with_name('bg-shared-model-blend-protocol.md')),
                                   'cases': report, 'upper_count': sum(not row['excluded_by_projection'] for row in report.values()),
                                   'midpoint_pass_count': sum(row['midpoint_passed'] for row in report.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
