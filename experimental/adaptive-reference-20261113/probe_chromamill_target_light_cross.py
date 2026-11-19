"""Cross controlled target and reference capture lights under the same Film A edit."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_chromamill_capture_cross import LEVELS, LIGHTS, SCENES
from probe_middlebury_camera_channel_replay import difference
from probe_target_mirror_chroma import BASE, CHECKPOINT, checked_json, checked_rgb, json_to, panel, save
from spatial_render import SpatialEngine, finish_spatial

PARENT_SHA = 'ae4bdcfbaec10bf43c11b882138745ef02f86f58011d6102a43a666630e83789'
STATES = ('off', 'on')


def load_scene(source, scene, row):
    return {(light, level, state): checked_rgb(source / f'{scene}-{light}-{level}-{state}.png',
                                              row[f'{light}-{level}']['hashes'][state])
            for light, level, state in product(LIGHTS, LEVELS, STATES)}


def build_targets(frames):
    targets = {}
    for light in LIGHTS:
        original = frames[light, 241, 'off']
        expected = finish_spatial(original, frames[light, 241, 'on'], .8)
        whole = np.ones(original.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)['neutral_proxy'] for rgb in (original, frames[light, 241, 'on'])))
        targets[light] = (original, expected, whole, neutral)
    return targets


def generate(engine, source, folder, scene, row, targets, references):
    renders, summary = {}, {}
    for target_light, (original, expected, _, _) in targets.items():
        board = {}
        for (ref_light, level, state), guide in references.items():
            tag = f'{ref_light}-{level}-{state}'
            baseline = finish_spatial(original, engine.prepare(original, guide).apply(original), .8)
            if target_light == 'white':
                prior = checked_rgb(source / f'{scene}-{tag}-13.png', row['hashes'][tag])
                np.testing.assert_array_equal(baseline, prior)
            renders[target_light, ref_light, level, state] = baseline
            summary[f'{target_light}-{tag}'] = save(folder / f'{target_light}-{tag}.png', baseline)
            board[tag] = {'target': original, 'reference': guide, '13': baseline,
                          'known edit': expected if state == 'on' else original}
        panel(folder / f'{target_light}-comparison.png', board)
    return renders, summary


def measure(targets, renders, prior):
    common = np.logical_and(*(row[3] for row in targets.values()))
    styles, interactions = {}, {}
    for target_light, (original, expected, whole, neutral) in targets.items():
        for ref_light, level in product(LIGHTS, LEVELS):
            key = f'{target_light}-{ref_light}-{level}'
            results = {}
            for label, area in (('whole', whole), ('neutral', neutral), ('common', common)):
                if area.sum() < 256:
                    results[label] = None
                    continue
                grades = [renders[target_light, ref_light, level, state] for state in STATES]
                values = metrics(original, expected, *grades, area)
                if target_light == 'white' and label != 'common' and values != prior[label]['styles'][f'{ref_light}-{level}']:
                    raise ValueError('White-target baseline metrics changed')
                results[label] = {**values, 'joint_pass': values['passed'] and values['off_matches_original_mae'] <= 3}
            styles[key] = results
    for label, area in (('whole', targets['white'][2]), ('common', common)):
        if area.sum() < 256:
            continue
        interactions[label] = {}
        for level in LEVELS:
            edits = []
            for target_light, ref_light in product(LIGHTS, repeat=2):
                off, on = (renders[target_light, ref_light, level, state][area].astype(float) for state in STATES)
                edits.append(on - off)
            ww, wb, bw, bb = edits
            interactions[label][str(level)] = difference(bb - bw - wb + ww)
    return styles, interactions, int(common.sum())


@torch.inference_mode()
def run(out):
    source = BASE / 'chromamill-capture-cross-v1'
    parent = checked_json(source / 'results.json', PARENT_SHA)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT or parent['model_sha256'] != CHECKPOINT:
        raise ValueError('Changed 13 weights')
    scenes = {scene: load_scene(source, scene, parent['prepared'][scene]) for scene in SCENES}
    out.mkdir()
    cases = {}
    matched = []
    for scene, donor in (SCENES, SCENES[::-1]):
        targets = build_targets(scenes[scene])
        folder = out / scene
        folder.mkdir()
        prior = parent['cases'][scene]
        renders, hashes = generate(engine, source, folder, scene, prior, targets, scenes[donor])
        styles, interactions, common = measure(targets, renders, prior['regions'])
        for light, level in product(LIGHTS, LEVELS):
            row = styles[f'{light}-{light}-{level}']
            matched.append(all(row[label] and row[label]['joint_pass'] for label in ('whole', 'common')))
        cases[scene] = dict(donor=donor, styles=styles, interactions=interactions, common_pixels=common,
                            neutral_pixels={light: int(row[3].sum()) for light, row in targets.items()}, hashes=hashes)
    json_to(out / 'results.json', dict(parent_sha256=PARENT_SHA, checkpoint_sha256=CHECKPOINT,
                                      matched_passing=sum(matched), matched_count=len(matched), passed=all(matched), cases=cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
