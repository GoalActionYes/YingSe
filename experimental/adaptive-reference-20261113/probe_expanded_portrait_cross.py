"""Check the frozen expanded model on the matched editor-portrait switch."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from background_intervention import target_regions
from probe_frozen_candidate_development_crosscheck import grade
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_rawtherapee_portrait_cross import CHECKPOINT, CUCKOO, managed, score
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save
from spatial_render import SpatialEngine

BASELINE_SHA = 'a34c9c55ee98af11c548157738b9f96db3419f919ba41d592eb3662a6be1a08a'
STATES = ('off', 'on')
DIRECTIONS = {'cuckoo': 'mixed', 'mixed': 'cuckoo'}
ARMS = {'expanded': False, 'statistics-only': True}


def load_inputs(root, baseline_dir, frozen):
    folders = {'cuckoo': root / 'rawtherapee-portrait-toning-v1', 'mixed': baseline_dir}
    result = {}
    for name, folder in folders.items():
        images = {}
        for state in STATES:
            base = folder / (state if name == 'cuckoo' else 'mixed-author-' + state)
            path = base / 'image.png'
            if name == 'cuckoo' and digest(path) != CUCKOO[state]:
                raise ValueError('Changed archived author export')
            pixels = managed(path)
            if sha256(pixels.tobytes()).hexdigest() != frozen[name]['inputs_rgb_sha256'][state]:
                raise ValueError('Changed frozen working RGB')
            images[state] = pixels
        result[name] = images
    return result


def measurement_masks(target, expected, boxes):
    empty = np.zeros(target.shape[:2], dtype=bool)
    masks = {'whole': ~empty, 'neutral': np.logical_and.reduce(
        [target_regions(image, empty)['neutral_proxy'] for image in (target, expected)])}
    for region, (x0, y0, x1, y1) in boxes.items():
        mask = empty.copy()
        mask[y0:y1, x0:x1] = True
        masks[region] = mask
    return masks


@torch.inference_mode()
def run(root, baseline_dir, out):
    prior = checked_json(baseline_dir / 'results.json', BASELINE_SHA)
    region_path = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    if digest(region_path) != prior['region_sha256'] or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed frozen expanded-model input')
    images = load_inputs(root, baseline_dir, prior['results'])
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 checkpoint')
    initialize_replication_model(engine.model, engine.model.state_dict())
    weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Nonfinite expanded weights')
    engine.model.load_state_dict(weights, strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    predictions, artifacts = {}, {}
    for name, reference in DIRECTIONS.items():
        target = images[name]['off']
        baseline = {state: checked_rgb(baseline_dir / f'{name}-from-{reference}-{state}.png',
                                       prior['results'][name]['predictions'][state]) for state in STATES}
        predictions[name] = {'13': baseline}
        artifacts[name] = {}
        for arm, appearance_disabled in ARMS.items():
            prepared = {state: grade(engine, target, images[reference][state], baseline[state],
                                     disable_appearance=appearance_disabled)[0] for state in STATES}
            predictions[name][arm] = prepared
            artifacts[name][arm] = {state: save(out / f'{name}-{arm}-{state}.png', image)
                                    for state, image in prepared.items()}
    results = {}
    for name, arms in predictions.items():
        baseline = prior['results'][name]
        target = images[name]['off']
        expected = checked_rgb(baseline_dir / f'{name}-expected.png', baseline['expected_sha256'])
        boxes = json.loads(region_path.read_bytes()) if name == 'cuckoo' else {}
        masks = measurement_masks(target, expected, boxes)
        scores = {arm: score(target, expected, frames, masks) for arm, frames in arms.items()}
        if scores['13'] != baseline['scores']:
            raise ValueError('Changed frozen baseline score')
        results[name] = {'scores': scores, 'outputs': artifacts[name]}
    passed = {arm: all(case['scores'][arm][zone]['passed'] for case in results.values()
                       for zone in ('whole', 'neutral')) for arm in ARMS}
    json_to(out / 'results.json', {'parent_sha256': BASELINE_SHA, 'checkpoint_sha256': WEIGHT_SHA256,
                                   'script_sha256': digest(Path(__file__)), 'results': results, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'baseline_dir', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
