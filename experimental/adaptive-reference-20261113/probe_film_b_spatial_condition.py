"""Swap only the low-resolution target condition during Film B grid generation."""
import argparse
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints, verified
from probe_arri_shared_look import metrics
from probe_arri_two_looks import ASSIGNMENTS, CHECKPOINT
from probe_target_mirror_chroma import checked_json, json_to, save
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial
from training_checks import finite_output

SOURCE_SHA = 'e09286501decd0d41cf727bce645d9920298da166f6bdaf4bab20385d27d730e'
CROSS_SHA = 'fea85f0672c6da99d5734e8fb52034438122a03a71b7e493534c4eb38da07237'
STATES = ('off', 'film_b')


@torch.inference_mode()
def run(source, crossed, out):
    report = checked_json(source / 'results.json', SOURCE_SHA)
    baseline = checked_json(crossed / 'results.json', CROSS_SHA)
    images = {scene: endpoints(source, report, scene) for scene in ASSIGNMENTS}
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT or baseline['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Changed frozen checkpoint')
    api = engine.api
    prepared = {scene: [engine.analysis(image) for image in frames] for scene, frames in images.items()}
    histograms = {scene: [api['batch_histogram_1D'](api['srgb_to_ycbcr'](tensor)) for tensor in tensors]
                  for scene, tensors in prepared.items()}
    layouts = {scene: api['TF'].resize(tensors[0], (64, 64), antialias=True)
               for scene, tensors in prepared.items()}
    cases = {}
    out.mkdir()
    for name, row in baseline['cases'].items():
        scene, donor = row['target'], row['reference']
        target, style = images[scene]
        expected = finish_spatial(target, style, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)['neutral_proxy'] for rgb in images[scene]))
        if neutral.sum() < 256:
            raise ValueError(f'Insufficient neutral support: {scene}')
        samples = {}
        prefix = f'{scene}-from-{donor}'
        for layout, low in layouts.items():
            frames, hashes = [], {}
            for state, reference_hist in zip(STATES, histograms[donor], strict=True):
                conditions = torch.cat([histograms[scene][0], reference_hist], 1)
                grid = api['lowpass'](engine.model.apply_grid(low, conditions))
                finite_output(grid, (1, 13, 8, 16, 16))
                picture = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)
                if layout == scene:
                    prior = verified(crossed / f'{prefix}-{state}.png', row['output_sha256'][state])
                    np.testing.assert_array_equal(picture, prior)
                frames.append(picture)
                hashes[state] = save(out / f'{prefix}-layout-{layout}-{state}.png', picture)
            samples[layout] = {'whole': metrics(target, expected, *frames, whole),
                               'neutral': metrics(target, expected, *frames, neutral), 'sha256': hashes}
        cases[name] = {'target': scene, 'reference': donor, 'layout': samples}
    weld = 'sparks_4592'
    weld_responses = [all(regions[key]['projection'] > 0 for key in ('whole', 'neutral'))
                      for row in cases.values() if row['target'] == weld
                      for layout, regions in row['layout'].items() if layout != weld]
    passed = sum(all(regions[key]['passed'] for key in ('whole', 'neutral'))
                 for row in cases.values() for regions in row['layout'].values())
    json_to(out / 'results.json', {'cross_sha256': CROSS_SHA, 'checkpoint_sha256': CHECKPOINT,
                                  'weld_positive': sum(weld_responses), 'weld_swaps': len(weld_responses),
                                  'passing_combinations': passed, 'cases': cases})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('source', 'crossed', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
