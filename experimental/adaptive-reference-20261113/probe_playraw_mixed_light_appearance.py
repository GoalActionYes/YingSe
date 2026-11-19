"""Compare frozen reference appearance on the real masked color-edit switch."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from background_intervention import target_regions
from probe_frozen_candidate_development_crosscheck import CHECKPOINT, STRENGTH, WEIGHT, WEIGHT_SHA256, grade
from probe_playraw_mixed_light_transfer import BASE, PARENT, PARENT_SHA, WHITE, chroma, read, regions, stats
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save
from spatial_render import SpatialEngine, finish_spatial
from verify_development_inputs import manifest


STORE = BASE / 'playraw-mixed-light-transfer-v1'
STORE_SHA = '460cc3ba726141617b424438d4b7297b3003d1392090f9f1c551a91c85911197'


def image_measure(frames, expected, zones, white):
    change = chroma(frames[1]) - chroma(frames[0])
    zeros = np.zeros_like(change)
    scores = {label: stats(change, expected if label == 'author' else zeros, zone)
              for label, zone in zones.items()}
    protected = all(row['response_p95'] <= 1.5 for label, row in scores.items() if label != 'author')
    author = scores.get('author')
    passed = protected and (white or (.75 <= author['projection'] <= 1.25 and author['response_mae'] <= 3))
    return {'scores': scores, 'passed': passed}


def condition(out, engine, key, target, references, expected, zones, prior):
    baseline = [checked_rgb(STORE / f'{key}-{state}.png', prior['outputs'][state]['image_sha256'])
                for state in ('off', 'on')]
    control = image_measure(baseline, expected, zones, key == 'white')
    prior_scores = {'white': prior['scores']} if key == 'white' else prior['scores']
    if control['scores'] != prior_scores or control['passed'] != prior['passed']:
        raise ValueError('Previous scoring changed')
    rows = {'13': control}
    for arm, disabled in (('statistics', True), ('appearance', False)):
        frames = [grade(engine, target, guide, original, disable_appearance=disabled)[0]
                  for guide, original in zip(references, baseline, strict=True)]
        if np.array_equal(target, references[0]):
            np.testing.assert_array_equal(frames[0], target)
        rows[arm] = image_measure(frames, expected, zones, key == 'white') | {'hashes': [
            save(out / f'{key}-{arm}-{role}.png', frame) for role, frame in zip(('off', 'on'), frames)]}
    return rows


def run(out):
    torch.set_num_threads(4)
    history = checked_json(STORE / 'results.json', STORE_SHA)
    provenance = checked_json(PARENT / 'results.json', PARENT_SHA)
    original, _ = read(PARENT / 'off.tif', False)
    edited, mask = read(PARENT / 'on.tif', True)
    if ([sha256(x.tobytes()).hexdigest() for x in (original, edited)] != provenance['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != provenance['mask_pixels_sha256']):
        raise ValueError('Changed author images')
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed frozen weights')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    expected = chroma(finish_spatial(original, edited, STRENGTH)) - chroma(original)
    neutral = target_regions(original, np.zeros_like(mask, bool))['neutral_proxy']
    zones = {'author': mask >= 204, 'protected': mask == 0, 'neutral': neutral & (mask == 0)}
    h, w = mask.shape
    crop = np.s_[h//16:h-h//16, w//16:w-w//16]
    pairs = {'full': (original, edited), 'mirror': (original[:, ::-1], edited[:, ::-1]),
             'crop': (original[crop], edited[crop])}
    out.mkdir()
    cases = {}
    for name, inputs in pairs.items():
        if not history['cases'][name]['qualified']:
            raise ValueError(f'Previous reference was not qualified: {name}')
        cases[name] = condition(out, engine, name, original, inputs, expected, zones, history['cases'][name])
    profile = manifest(BASE, 'bg-real-editor-look')['cases'][WHITE]
    jersey = checked_rgb(BASE / 'bg-real-editor-look' / WHITE / 'target.png', profile['artifacts']['target.png'])
    white = np.zeros(jersey.shape[:2], bool)
    white[regions()[WHITE]['白球服']] = True
    cases['white'] = condition(out, engine, 'white', jersey, pairs['full'], None, {'white': white}, history['white'])
    gain = any(rows['appearance']['scores']['author']['projection'] > rows['statistics']['scores']['author']['projection']
               for key, rows in cases.items() if key != 'white')
    passed = all(row['appearance']['passed'] for row in cases.values()) and gain
    json_to(out / 'results.json', {'parent_sha256': STORE_SHA, 'checkpoint_sha256': WEIGHT_SHA256,
                                   'cases': cases, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
