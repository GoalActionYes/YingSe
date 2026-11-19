"""Render a chroma correction from the same two images and a reverse grade."""

import argparse
from pathlib import Path

import numpy as np
import torch

from probe_frozen_candidate_development_crosscheck import COUNTS, film_b
from probe_reference_black_anchor import evaluate_regions
from probe_reference_skin_chroma_isolation import skin_boxes
from probe_stable_chroma_hybrid import chart
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA, LUMA_INTEGER, STRENGTH, SpatialEngine, background, chroma, facd,
    json_to, render, save,
)
from spatial_render import finish_spatial
from verify_development_inputs import MANIFESTS, verify


def apply_correction(engine, reverse_folder, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'route': 'same_image'}
    if np.array_equal(target, baseline):
        return baseline.copy(), {'route': 'no_forward_change'}
    reverse = finish_spatial(baseline, engine.prepare(baseline, target).apply(baseline), STRENGTH)
    drift = reverse.astype(float) - target
    residual = chroma(drift, drift @ LUMA)
    y = baseline @ LUMA
    estimate, limited = render(chroma(baseline, y) - .5 * residual, y)
    luma_error = np.abs((estimate.astype(np.int32) - baseline) @ LUMA_INTEGER) / 10000
    if luma_error.max() > .5:
        raise ValueError('Unexpected luma change outside rounding')
    name = f'{len(list(reverse_folder.iterdir())):03d}.png'
    reverse_sha = save(reverse_folder / name, reverse)
    return estimate, {'route': 'corrected', 'reverse_file': name, 'reverse_sha256': reverse_sha,
                      'residual_mae': float(np.abs(residual).mean()),
                      'changed_pixels': int(np.any(estimate != baseline, axis=-1).sum()),
                      'max_luma_error': float(luma_error.max()), 'gamut_limited_fraction': limited}


@torch.inference_mode()
def run(out):
    checked = verify(BASE, COUNTS)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed frozen 13 checkpoint')
    out.mkdir()
    reverse_folder = out / 'reverse'
    reverse_folder.mkdir()
    sample = np.full((32, 32, 3), 100, np.uint8)
    other = 255 - sample
    for reference, baseline, route in ((sample.copy(), other, 'same_image'), (other, sample.copy(), 'no_forward_change')):
        result, debug = apply_correction(engine, reverse_folder, sample, reference, baseline)
        np.testing.assert_array_equal(result, sample)
        if debug['route'] != route:
            raise ValueError('Identity guard changed')
    def candidate(runner, original, guide, baseline):
        return apply_correction(runner, reverse_folder, original, guide, baseline)

    suites = {name: check(engine, out / name, candidate, 'Chroma round trip')
              for name, check in (('background', background), ('facd', facd), ('film_b', film_b), ('chart', chart))}
    rows = skin_boxes(out, suites['background'])
    regions = evaluate_regions(rows)
    json_to(out / 'results.json', dict(baseline_sha256=CHECKPOINT, manifests=MANIFESTS, inputs=checked,
                                      suites=suites, regions=rows, region_gate=regions,
                                      passed=regions['passed'] and all(row['passed'] for row in suites.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
