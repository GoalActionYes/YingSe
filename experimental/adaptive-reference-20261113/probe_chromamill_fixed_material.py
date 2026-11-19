"""Measure fixed materials and replay the predicted edit on an ideal off endpoint."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_chromamill_target_light_cross import LEVELS, LIGHTS, PARENT_SHA, SCENES, STATES, load_scene
from probe_middlebury_camera_channel_replay import difference
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, panel, save
from spatial_render import finish_spatial

CROSS_SHA = '4542077d41de07bd7462db505d418d190ce81d0912fa24181fe7f155884230ad'
BOXES = {
    'Scene1_a': ((232, 12, 281, 30), (446, 156, 487, 185), (346, 296, 387, 322), (213, 229, 242, 252)),
    'Scene2_a': ((36, 19, 64, 51), (349, 12, 390, 35), (330, 302, 375, 324), (442, 152, 463, 178)),
}
SURFACES = ('surface_1', 'surface_2', 'surface_3')


def areas_for(original, scene):
    areas = {'whole': np.s_[:, :]}
    h, w = original.shape[:2]
    for label, (left, top, right, bottom) in zip((*SURFACES, 'colored_control'), BOXES[scene], strict=True):
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            raise ValueError('Fixed material outside image')
        areas[label] = np.s_[top:bottom, left:right]
    return areas


def score(original, expected, off, on, area):
    stats = metrics(original, expected, off, on, area)
    for key, predicted, truth in (('off', off, original), ('on', on, expected)):
        delta = predicted[area].astype(float) - truth[area]
        stats[f'{key}_color_error'] = difference(delta.reshape(-1, 3))
    return stats


def process(folder, prior, scene, originals, receipt):
    folder.mkdir()
    styles = {}
    for t in LIGHTS:
        original = originals[t, 241, 'off']
        expected = finish_spatial(original, originals[t, 241, 'on'], .8)
        areas = areas_for(original, scene)
        board = {}
        for r, level in product(LIGHTS, LEVELS):
            key = f'{t}-{r}-{level}'
            off, on = [checked_rgb(prior / scene / f'{key}-{state}.png', receipt['hashes'][f'{key}-{state}'])
                       for state in STATES]
            exact_unclipped = original.astype(np.int16) + on.astype(np.int16) - off
            exact = np.clip(exact_unclipped, 0, 255).astype(np.uint8)
            artifact = save(folder / f'{key}-ideal-base.png', exact)
            by_area = {label: {'baseline': score(original, expected, off, on, area),
                               'ideal_base': score(original, expected, original, exact, area),
                               'clipped_fraction': float(np.any(exact_unclipped[area] != exact[area], axis=-1).mean())}
                       for label, area in areas.items()}
            previous = receipt['styles'][key]['whole']
            if any(by_area['whole']['baseline'][k] != v for k, v in previous.items() if k != 'joint_pass'):
                raise ValueError('Baseline whole-image readings changed')
            required = ('whole', *SURFACES)
            passes = all(by_area[label]['ideal_base']['passed'] for label in required)
            styles[key] = dict(areas=by_area, photo_sha256=artifact, passes=passes, matched=t == r)
            board[f'{r}-{level}'] = {'target': original, '13 on': on, 'ideal off plus 13 edit': exact, 'known edit': expected}
        panel(folder / f'{t}-comparison.png', board)
    return styles


def run(out):
    prior = BASE / 'chromamill-target-light-cross-v2'
    source = BASE / 'chromamill-capture-cross-v1'
    cross = checked_json(prior / 'results.json', CROSS_SHA)
    parent = checked_json(source / 'results.json', PARENT_SHA)
    if cross['parent_sha256'] != PARENT_SHA or not all(row['qualified'] for row in parent['checks'].values()):
        raise ValueError('Missing qualified aligned capture pairs')
    out.mkdir()
    cases = {scene: process(out / scene, prior, scene, load_scene(source, scene, parent['prepared'][scene]), cross['cases'][scene])
             for scene in SCENES}
    matched = [row['passes'] for scene in cases.values() for row in scene.values() if row['matched']]
    if len(matched) != 8:
        raise ValueError('Expected eight matched conditions')
    json_to(out / 'results.json', dict(parent_sha256=CROSS_SHA, boxes=BOXES, cases=cases,
                                      matched_passing=sum(matched), passes=any(matched)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
