"""Try luma-only reference appearance with unchanged RGB color conditions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_expanded_portrait_cross import BASELINE_SHA, DIRECTIONS, load_inputs, measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, STRENGTH
from probe_portrait_reference_paths import EXPANDED_SHA
from probe_rawtherapee_portrait_cross import CHECKPOINT, score
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


def render(engine, target, reference, luma_only):
    if np.array_equal(target, reference):
        return target.copy()
    target_input = engine.analysis(target)
    reference_input = engine.analysis(reference)
    appearance = reference_input
    if luma_only:
        luma_weights = reference_input.new_tensor(LUMA).view(1, 3, 1, 1)
        luma = (reference_input * luma_weights).sum(dim=1, keepdim=True).clamp(0, 1)
        appearance = luma.expand_as(reference_input)
    grid = appearance_grid(engine.model, engine.api, target_input, reference_input,
                           appearance_reference=appearance)
    return finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), STRENGTH)


@torch.inference_mode()
def run(root, baseline_dir, expanded_dir, out):
    base = checked_json(baseline_dir / 'results.json', BASELINE_SHA)
    expanded = checked_json(expanded_dir / 'results.json', EXPANDED_SHA)
    if expanded['checkpoint_sha256'] != WEIGHT_SHA256 or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed expanded weights')
    regions = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    if digest(regions) != base['region_sha256']:
        raise ValueError('Changed measurement regions')
    boxes = json.loads(regions.read_bytes())
    images = load_inputs(root, baseline_dir, base['results'])
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 base weights')
    initialize_replication_model(engine.model, engine.model.state_dict())
    weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('Nonfinite expanded weights')
    engine.model.load_state_dict(weights, strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir()
    predictions = {}
    for name, reference in DIRECTIONS.items():
        target = images[name]['off']
        np.testing.assert_array_equal(render(engine, target, target.copy(), True), target)
        frames, hashes = {}, {}
        for state, rgb in images[reference].items():
            source = expanded['results'][name]['outputs']['expanded'][state]
            archived = checked_rgb(expanded_dir / f'{name}-expanded-{state}.png', source)
            np.testing.assert_array_equal(render(engine, target, rgb, False), archived)
            frames[state] = render(engine, target, rgb, True)
            hashes[state] = save(out / f'{name}-{state}.png', frames[state])
        predictions[name] = (frames, hashes)
    results = {}
    for name, (frames, hashes) in predictions.items():
        target = images[name]['off']
        source = base['results'][name]
        expected = checked_rgb(baseline_dir / f'{name}-expected.png', source['expected_sha256'])
        masks = measurement_masks(target, expected, boxes if name == 'cuckoo' else {})
        readings = score(target, expected, frames, masks)
        results[name] = {'scores': readings, 'sha256': hashes,
                         'passed': all(readings[region]['passed'] for region in ('whole', 'neutral'))}
    json_to(out / 'results.json', {'parent_sha256': EXPANDED_SHA, 'checkpoint_sha256': WEIGHT_SHA256,
                                  'source_sha256': digest(Path(__file__)), 'identity_exact': True,
                                  'results': results, 'passed': all(row['passed'] for row in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'baseline_dir', 'expanded_dir', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
