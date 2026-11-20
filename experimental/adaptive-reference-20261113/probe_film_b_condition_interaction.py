"""Measure RGB-response interaction between the two target conditions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_film_b_spatial_condition import (
    ASSIGNMENTS, CHECKPOINT, SOURCE_SHA, STATES, PreparedSpatial, SpatialEngine,
    checked_json, endpoints, finite_output, finish_spatial, json_to, metrics, save, target_regions, verified,
)

HISTOGRAM_SHA = 'e70732021df6ec9d004ff7cf50d7aa590850df8ff80a917ee1085f9c03f3601d'
LAYOUT_SHA = '12844c46bd6ef80f4784adf4ed8196eaf129a1c84b939afda2f0171436f7067c'


def read_pair(folder, prefix, hashes):
    return [verified(folder / f'{prefix}-{state}.png', hashes[state]) for state in STATES]


def difference(pair):
    off, on = pair
    return on.astype(np.int16) - off


def interaction(target, expected, actual, additive, mask, scores):
    wanted = expected[mask].astype(np.float64) - target[mask]
    predicted = additive[mask]
    error = actual[mask] - predicted
    energy = np.square(wanted).sum()
    additive_projection = float((predicted * wanted).sum() / energy)
    gap = scores['projection'] - additive_projection
    return {**scores, 'additive_projection': additive_projection, 'projection_gap': gap,
            'interaction_mae': float(np.abs(error).mean()),
            'sign_changed': (scores['projection'] > 0) != (additive_projection > 0),
            'nonadditive': abs(gap) >= .15 or float(np.abs(error).mean()) >= .5}


@torch.inference_mode()
def run(source, histogram_dir, layout_dir, out):
    source_report = checked_json(source / 'results.json', SOURCE_SHA)
    hist_report = checked_json(histogram_dir / 'results.json', HISTOGRAM_SHA)
    layout_report = checked_json(layout_dir / 'results.json', LAYOUT_SHA)
    images = {scene: endpoints(source, source_report, scene) for scene in ASSIGNMENTS}
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed model')
    api = engine.api
    prepared = {scene: [engine.analysis(image) for image in frames] for scene, frames in images.items()}
    histograms = {scene: [api['batch_histogram_1D'](api['srgb_to_ycbcr'](tensor)) for tensor in tensors]
                  for scene, tensors in prepared.items()}
    layouts = {scene: api['TF'].resize(tensors[0], (64, 64), antialias=True)
               for scene, tensors in prepared.items()}
    results = {}
    out.mkdir()
    for name, layout_case in layout_report['cases'].items():
        scene, donor = layout_case['target'], layout_case['reference']
        stats_case = hist_report['cases'][name]['statistics']
        target, style = images[scene]
        expected = finish_spatial(target, style, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(rgb, ~whole)['neutral_proxy'] for rgb in images[scene]))
        prefix = f'{scene}-from-{donor}'
        samples = tuple(key for key in images if key != scene)
        own = read_pair(layout_dir, f'{prefix}-layout-{scene}', layout_case['layout'][scene]['sha256'])
        historical = read_pair(histogram_dir, f'{prefix}-stats-{scene}', stats_case[scene]['output_sha256'])
        for first, second in zip(own, historical, strict=True):
            np.testing.assert_array_equal(first, second)
        base_delta = difference(own)
        stats_delta = {k: difference(read_pair(histogram_dir, f'{prefix}-stats-{k}',
                                             stats_case[k]['output_sha256'])) for k in samples}
        layout_delta = {k: difference(read_pair(layout_dir, f'{prefix}-layout-{k}',
                                              layout_case['layout'][k]['sha256'])) for k in samples}
        combinations = {}
        for stats in samples:
            for layout in samples:
                frames, hashes = [], {}
                for state, reference_hist in zip(STATES, histograms[donor], strict=True):
                    conditions = torch.cat([histograms[stats][0], reference_hist], 1)
                    grid = api['lowpass'](engine.model.apply_grid(layouts[layout], conditions))
                    finite_output(grid, (1, 13, 8, 16, 16))
                    rgb = finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)
                    frames.append(rgb)
                    hashes[state] = save(out / f'{prefix}-stats-{stats}-layout-{layout}-{state}.png', rgb)
                additive = stats_delta[stats] + layout_delta[layout] - base_delta
                actual = difference(frames)
                rows = {label: interaction(target, expected, actual, additive, mask,
                                           metrics(target, expected, *frames, mask))
                        for label, mask in (('whole', whole), ('neutral', neutral))}
                combinations[f'{stats}+{layout}'] = dict(regions=rows, sha256=hashes)
        results[name] = dict(target=scene, reference=donor, combinations=combinations)
    rows = [entry['regions'] for row in results.values() for entry in row['combinations'].values()]
    weld = [entry['regions'] for row in results.values() if row['target'] == 'sparks_4592'
            for entry in row['combinations'].values()]
    json_to(out / 'results.json', dict(histogram_sha256=HISTOGRAM_SHA, layout_sha256=LAYOUT_SHA,
                                      number=len(rows), weld_nonadditive=sum(any(y['nonadditive'] for y in x.values()) for x in weld),
                                      sign_changes=sum(any(y['sign_changed'] for y in x.values()) for x in rows),
                                      passing=sum(all(y['passed'] for y in x.values()) for x in rows), cases=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'histogram_dir', 'layout_dir', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
