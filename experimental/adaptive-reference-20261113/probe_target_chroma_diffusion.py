"""Test edge-guided redistribution of the frozen grade's chroma residual."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_chroma_round_trip import (
    BASE, CHECKPOINT, COUNTS, LUMA, LUMA_INTEGER, MANIFESTS, SpatialEngine, background, chart, chroma,
    evaluate_regions, facd, film_b, json_to, render, skin_boxes, verify,
)


def redistribute(source, residual, stride):
    height, width = source.shape[:2]
    edges = []
    for axis in (0, 1):
        if source.shape[axis] <= stride:
            continue
        ends = ((np.s_[:-stride, :], np.s_[stride:, :]) if axis == 0
                else (np.s_[:, :-stride], np.s_[:, stride:]))
        first, second = ends
        distance = np.square(source[first] - source[second]).sum(axis=-1)
        edges.append((first, second, np.exp(-distance / (2 * 12.**2))[..., None] / 8))
    estimate = residual.copy()
    for _ in range(4):
        previous = estimate.copy()
        for first, second, weight in edges:
            change = weight * (previous[second] - previous[first])
            estimate[first] += change
            estimate[second] -= change
    return estimate


def candidate(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'route': 'same_image'}
    if np.array_equal(target, baseline):
        return baseline.copy(), {'route': 'no_forward_change'}
    source = target.astype(float)
    source_chroma = chroma(source, source @ LUMA)
    output_y = baseline @ LUMA
    output_chroma = chroma(baseline, output_y)
    old = output_chroma - source_chroma
    stride = max(1, round(min(target.shape[:2]) / 32))
    shifted = redistribute(source, old, stride)
    np.testing.assert_allclose(shifted.mean(axis=(0, 1)), old.mean(axis=(0, 1)), atol=1e-10, rtol=0)
    rgb, limited = render(source_chroma + shifted, output_y)
    luma_error = float(np.abs((rgb.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000)
    if luma_error > .5:
        raise ValueError('Unexpected luma change beyond quantization')
    actual_chroma = chroma(rgb, rgb @ LUMA)
    return rgb, {'route': 'redistributed', 'stride': stride, 'max_luma_error': luma_error,
                 'change_mae': float(np.abs(actual_chroma - output_chroma).mean()),
                 'mean_chroma_change': (actual_chroma - output_chroma).mean(axis=(0, 1)).tolist(),
                 'gamut_limited_fraction': limited}


@torch.inference_mode()
def run(out):
    verified = verify(BASE, COUNTS)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed frozen baseline checkpoint')
    out.mkdir()
    suites = {name: check(engine, out / name, candidate, 'Target edge chroma diffusion')
              for name, check in (('background', background), ('facd', facd), ('film_b', film_b), ('chart', chart))}
    regions = skin_boxes(out, suites['background'])
    region_gate = evaluate_regions(regions)
    json_to(out / 'results.json', dict(baseline_sha256=CHECKPOINT, manifests=MANIFESTS, inputs=verified,
                                      suites=suites, regions=regions, region_gate=region_gate,
                                      passed=region_gate['passed'] and all(row['passed'] for row in suites.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
