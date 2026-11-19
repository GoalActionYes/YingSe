"""Replay known edit changes across calibrated capture lights for the same materials."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_chromamill_fixed_material import SURFACES, areas_for
from probe_chromamill_target_light_cross import LEVELS, PARENT_SHA, SCENES, load_scene
from probe_physical_blue_global_fit import display, inside, linear
from probe_target_mirror_chroma import BASE, checked_json, json_to, panel, save
from spatial_render import finish_spatial

GAIN = np.array([.8, .9, .85])
DIRECTIONS = (('white', 'blue'), ('blue', 'white'))


def replay(before, edited, target, supported):
    base = linear(target)
    origin = linear(before)[supported]
    delta = linear(edited)[supported] - origin
    ratio = base[supported] / origin
    arms = {}
    for key, change in (('direct', delta), ('scaled', delta * ratio)):
        estimate = base[supported] + change
        unblended = target.copy()
        unblended[supported] = display(estimate)
        output = finish_spatial(target, unblended, .8)
        clipped = np.any((estimate < 0) | (estimate > 1), axis=-1)
        arms[key] = (output, float(clipped.mean()))
    return arms


def evaluate(target, expected, arms, supported, areas):
    scores = {key: {} for key in arms}
    sizes = {}
    for name, area in areas.items():
        mask = supported[area]
        sizes[name] = {'pixels': int(mask.sum()), 'fraction': float(mask.mean())}
        qualified = mask.sum() >= 256 and (name != 'whole' or mask.mean() >= .90)
        for key, (frame, _) in arms.items():
            scores[key][name] = (metrics(target[area], expected[area], target[area], frame[area], mask)
                                 if qualified else None)
    for key in scores:
        scores[key]['passes'] = all(scores[key][a] and scores[key][a]['passed'] for a in ('whole', *SURFACES))
    return scores, sizes


def measure(folder, frames, scene):
    folder.mkdir()
    cases, panels = {}, {}
    for (donor, dest), level in product(DIRECTIONS, LEVELS):
        before, edited = (frames[donor, level, state] for state in ('off', 'on'))
        target, authored = (frames[dest, level, state] for state in ('off', 'on'))
        support = inside(before) & inside(target)
        if not np.any(support):
            raise ValueError('No valid light-ratio support')
        expected = finish_spatial(target, authored, .8)
        key = f'{donor}-to-{dest}-{level}'
        areas = areas_for(target, scene)
        versions = replay(before, edited, target, support)
        gain_before = display(linear(before) * GAIN)
        gain_target = finish_spatial(target, display(linear(target) * GAIN), .8)
        controls = replay(before, gain_before, target, support)
        scores, pixels = evaluate(target, expected, versions, support, areas)
        digital, _ = evaluate(target, gain_target, controls, support, areas)
        frames_for_panel = {'target': target}
        artifacts = {}
        for name, (image, clipped) in versions.items():
            artifacts[name] = dict(sha256=save(folder / f'{key}-{name}.png', image), clipping=clipped)
            frames_for_panel[name] = image
        frames_for_panel['expected'] = expected
        panels[key] = frames_for_panel
        cases[key] = dict(scores=scores, support=pixels, artifacts=artifacts, digital_gain=digital)
    board = panel(folder / 'comparison.png', panels)
    return dict(cases=cases, panel_sha256=board)


def run(out):
    source = BASE / 'chromamill-capture-cross-v1'
    parent = checked_json(source / 'results.json', PARENT_SHA)
    if not all(row['qualified'] for row in parent['checks'].values()):
        raise ValueError('Unqualified physical capture')
    out.mkdir()
    scenes = {scene: measure(out / scene, load_scene(source, scene, parent['prepared'][scene]), scene) for scene in SCENES}
    cases = [case for scene in scenes.values() for case in scene['cases'].values()]
    counts = {arm: sum(row['scores'][arm]['passes'] for row in cases) for arm in ('direct', 'scaled')}
    controls = sum(row['digital_gain']['scaled']['passes'] for row in cases)
    errors = [row['scores']['scaled']['whole']['response_mae'] - row['scores']['direct']['whole']['response_mae']
              for row in cases if row['scores']['scaled']['whole'] is not None]
    worst = max(errors, default=None)
    passed = (controls == len(cases) == 8 and counts['scaled'] >= 6 and counts['scaled'] > counts['direct']
              and worst is not None and worst <= 2)
    json_to(out / 'results.json', dict(parent_sha256=PARENT_SHA, counts=counts, digital_pass=controls,
                                      worst_regression=worst, passed=passed, scenes=scenes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
