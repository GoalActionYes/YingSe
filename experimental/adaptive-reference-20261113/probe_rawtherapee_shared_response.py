"""Test an RGB-dependent chroma response on whole held-out editor pairs."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import BASELINE_SHA, STATES, load_inputs, measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_portrait_cross import STRENGTH, managed, score
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, json_to, save
from spatial_render import finish_spatial

CUE_SHA = '971d8ef7b0132f97ab69203854ef72a35f2ee73e97e0e18d45ae2e3d983ff5f9'
REGION_SHA = '4e9e6f405312cfe01185095086b7f2629945b7cd0c07cd346d67c2b7a7e8e098'
SKIES = ('P1005231', 'R6_06283')
SAMPLES = 8192


def features(rgb):
    light = (rgb @ LUMA) / 255.
    dark = np.maximum(1. - 2. * light, 0.)
    bright = np.maximum(2. * light - 1., 0.)
    bands = np.stack((dark, 1. - dark - bright, bright), axis=-1)
    color = chroma(rgb) / 128.
    terms = np.column_stack((np.ones(len(rgb)), color))
    return (bands[:, :, None] * terms[:, None, :]).reshape(len(rgb), 9)


def training_pair(states):
    before, after = (states[state] for state in STATES)
    expected = finish_spatial(before, after, STRENGTH)
    indices = np.linspace(0, before.shape[0] * before.shape[1] - 1, SAMPLES, dtype=int)
    source = before.reshape(-1, 3)[indices]
    edited = expected.reshape(-1, 3)[indices]
    return features(source), chroma(edited) - chroma(source)


def fit(pairs):
    design, changes = (np.concatenate(parts) for parts in zip(*pairs))
    penalty = np.eye(design.shape[1]) * (len(design) * .001)
    return changes.mean(0), np.linalg.solve(design.T @ design + penalty, design.T @ changes)


def run(root, cue_report, out):
    baseline_dir = root / 'rawtherapee-portrait-cross-v1'
    baseline = checked_json(baseline_dir / 'results.json', BASELINE_SHA)
    registered = checked_json(Path(__file__).with_name('rawtherapee-portrait-toning-regions.json'), REGION_SHA)
    cue = checked_json(cue_report, CUE_SHA)
    images = load_inputs(root, baseline_dir, baseline['results'])
    for name in SKIES:
        images[name] = {state: managed(root / 'rawtherapee-local-edit-cue-v1'
                                      / f'{name}-{state}' / 'image.png') for state in STATES}
    for name, states in images.items():
        if any(sha256(rgb.tobytes()).hexdigest() != cue['inputs'][name]['rgb_sha256'][state]
               for state, rgb in states.items()):
            raise ValueError(f'Changed paired working RGB for {name}')
    out.mkdir()
    pairs = {name: training_pair(states) for name, states in images.items()}
    report = {}
    for name, states in images.items():
        donors = [group for group in pairs if group != name]
        plain, weights = fit([pairs[group] for group in donors])
        before, after = (states[state] for state in STATES)
        y = before @ LUMA
        colors = chroma(before)
        adaptive = (features(before.reshape(-1, 3)) @ weights).reshape(colors.shape)
        expected = finish_spatial(before, after, STRENGTH)
        masks = measurement_masks(before, expected, registered if name == 'cuckoo' else {})
        measured_masks = {region: mask for region, mask in masks.items() if mask.any()}
        arms = {}
        for arm, vector in (('constant', plain), ('pixel-color', adaptive)):
            prediction, limited = render(colors + vector, y)
            arms[arm] = {
                'prediction_sha256': save(out / f'{name}-{arm}.png', prediction),
                'gamut_limited_fraction': limited,
                'regions': score(before, expected, {'off': before, 'on': prediction}, measured_masks),
            }
        report[name] = {
            'donors': donors,
            'expected_sha256': save(out / f'{name}-expected.png', expected),
            'arms': arms,
            'empty_regions': sorted(set(masks) - set(measured_masks)),
        }
    adaptive_rows = {name: item['arms']['pixel-color']['regions'] for name, item in report.items()}
    regressions = [
        [name, region]
        for name, rows in adaptive_rows.items()
        for region, row in rows.items()
        if row['qualified'] and row['response_mae'] >
        report[name]['arms']['constant']['regions'][region]['response_mae'] + .25
    ]
    portraits_pass = all(
        adaptive_rows[name]['whole']['passed'] and
        all(row['passed'] for row in adaptive_rows[name].values() if row['qualified'])
        for name in ('cuckoo', 'mixed')
    )
    sky_pass = any(adaptive_rows[name]['whole']['passed'] for name in SKIES)
    json_to(out / 'results.json', {'cue_sha256': CUE_SHA, 'region_sha256': REGION_SHA, 'cases': report,
                                  'regressions': regressions,
                                  'passed': portraits_pass and sky_pass and not regressions})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'cue_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
