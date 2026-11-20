"""Isolate the two reference signals on the existing editor portrait switch."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_expanded_portrait_cross import BASELINE_SHA, DIRECTIONS, load_inputs, measurement_masks
from probe_playraw_mixed_light_transfer import STRENGTH, chroma
from probe_rawtherapee_portrait_cross import CHECKPOINT, score
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial

EXPANDED_SHA = '4851a926cee69eee76557680385a828603efe1b92465b829f6493c45139957ee'
COMBINATIONS = {'both-off': ('off', 'off'), 'both-on': ('on', 'on'),
                'statistics-on': ('on', 'off'), 'appearance-on': ('off', 'on')}


def forward(engine, target, reference):
    content = engine.analysis(target)
    source = {state: engine.analysis(image) for state, image in reference.items()}
    output = {}
    for name, (colors, appearance) in COMBINATIONS.items():
        grid = appearance_grid(engine.model, engine.api, content, source[colors],
                               appearance_reference=source[appearance])
        result = PreparedSpatial(engine, grid, {}).apply(target)
        output[name] = finish_spatial(target, result, STRENGTH)
    return output


def interaction(target, expected, frames, masks):
    actual = {name: chroma(frame) for name, frame in frames.items()}
    residual = actual['both-on'] - actual['appearance-on'] - actual['statistics-on'] + actual['both-off']
    desired = chroma(expected) - chroma(target)
    rows = {}
    for name, mask in masks.items():
        observed, want = residual[mask], desired[mask]
        energy = np.square(want).sum()
        rows[name] = {'chroma_mae': float(np.abs(observed).mean()),
                      'projection': float(np.sum(observed * want) / energy) if energy else None}
    return rows


@torch.inference_mode()
def run(root, baseline_dir, expanded_dir, out):
    base = checked_json(baseline_dir / 'results.json', BASELINE_SHA)
    expanded = checked_json(expanded_dir / 'results.json', EXPANDED_SHA)
    if expanded['checkpoint_sha256'] != WEIGHT_SHA256 or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed expanded weights')
    region_file = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    if digest(region_file) != base['region_sha256']:
        raise ValueError('Changed measurement regions')
    boxes = json.loads(region_file.read_bytes())
    images = load_inputs(root, baseline_dir, base['results'])
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed base weights')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    grades, hashes = {}, {}
    for name, reference in DIRECTIONS.items():
        frames = forward(engine, images[name]['off'], images[reference])
        outputs = expanded['results'][name]['outputs']['expanded']
        for state in ('off', 'on'):
            archived = checked_rgb(expanded_dir / f'{name}-expanded-{state}.png', outputs[state])
            np.testing.assert_array_equal(frames['both-' + state], archived)
        output_hashes = {option: save(out / f'{name}-{option}.png', value) for option, value in frames.items()}
        grades[name], hashes[name] = frames, output_hashes
    results = {}
    for name, frames in grades.items():
        target = images[name]['off']
        expected = checked_rgb(baseline_dir / f'{name}-expected.png', base['results'][name]['expected_sha256'])
        masks = measurement_masks(target, expected, boxes if name == 'cuckoo' else {})
        rows = {arm: score(target, expected, {'off': frames['both-off'], 'on': frames[arm]}, masks)
                for arm in ('both-on', 'statistics-on', 'appearance-on')}
        if rows['both-on'] != expanded['results'][name]['scores']['expanded']:
            raise ValueError('Changed expanded model scores')
        results[name] = {'scores': rows, 'interaction': interaction(target, expected, frames, masks),
                         'output_sha256': hashes[name]}
    json_to(out / 'results.json', {'base_sha256': BASELINE_SHA, 'expanded_sha256': EXPANDED_SHA,
                                   'checkpoint_sha256': WEIGHT_SHA256, 'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'baseline_dir', 'expanded_dir', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
