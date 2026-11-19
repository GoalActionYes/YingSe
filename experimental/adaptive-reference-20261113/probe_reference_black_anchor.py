"""Test a near-black chroma anchor in the existing two-image photo suites."""

import argparse
from pathlib import Path

import numpy as np
import torch

from probe_frozen_candidate_development_crosscheck import COUNTS, film_b
from probe_reference_skin_chroma_attribution import CORE, enough
from probe_reference_skin_chroma_isolation import skin_boxes
from probe_stable_chroma_hybrid import chart
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA, LUMA_INTEGER, SpatialEngine, background, chroma, facd, json_to, render,
)
from verify_development_inputs import MANIFESTS, verify


def support(rgb):
    y = rgb @ LUMA
    return (y >= 4) & (y <= 32) & (rgb.min(axis=-1) > 0) & (rgb.max(axis=-1) < 255)


def apply_anchor(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'route': 'same_image'}
    target_pixels, guide_pixels = support(target), support(reference)
    counts = {'target_support': int(target_pixels.sum()), 'reference_support': int(guide_pixels.sum())}
    if min(counts.values()) < 256:
        return baseline.copy(), {'route': 'low_support', **counts}
    old_y = baseline @ LUMA
    old_colors = chroma(baseline, old_y)
    guide = reference[guide_pixels]
    raw = np.median(chroma(guide, guide @ LUMA), axis=0) - np.median(old_colors[target_pixels], axis=0)
    offset = np.clip(raw, -8., 8.)
    output, limited = render(old_colors + offset, old_y)
    error = np.abs((output.astype(np.int32) - baseline) @ LUMA_INTEGER) / 10000
    if error.max() > .5:
        raise ValueError('Unexpected luminance change')
    return output, {'route': 'anchored', **counts, 'raw_offset': raw.tolist(), 'offset': offset.tolist(),
                    'max_luma_error': float(error.max()), 'gamut_limited_fraction': limited}


def evaluate_regions(rows):
    improved, regressions = 0, []
    for pair, areas in rows.items():
        for name, states in areas.items():
            for state, values in states.items():
                before, after = values['baseline'], values['candidate']
                if name in CORE.get(pair, ()):
                    improved += int(enough(before, after))
                elif after - before > .25:
                    regressions.append(dict(pair=pair, region=name, state=state, increase=after - before))
    return dict(improved=improved, regressions=regressions, passed=improved >= 5 and not regressions)


def paired_activation(suites):
    for state in ('C1', 'F1'):
        for row in suites['background']['cases'].values():
            if all(row['outputs'][f'{state}-{name}']['route'] == 'anchored' for name in ('off', 'on')):
                return True
    return False


@torch.inference_mode()
def run(out):
    inputs = verify(BASE, COUNTS)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('13 checkpoint changed')
    flat = np.full((64, 64, 3), 128, np.uint8)
    other = 255 - flat
    np.testing.assert_array_equal(apply_anchor(engine, flat, flat.copy(), other)[0], flat)
    fallback, detail = apply_anchor(engine, flat, other, other)
    np.testing.assert_array_equal(fallback, other)
    if detail['route'] != 'low_support':
        raise ValueError('Flat negative did not use the fallback')
    out.mkdir()
    suites = {name: run_suite(engine, out / name, apply_anchor, 'Near-black anchor')
              for name, run_suite in (('background', background), ('facd', facd), ('film_b', film_b), ('chart', chart))}
    rows = skin_boxes(out, suites['background'])
    regions = evaluate_regions(rows)
    activated = paired_activation(suites)
    json_to(out / 'results.json', dict(checkpoint_sha256=CHECKPOINT, input_manifests=MANIFESTS, preflight=inputs,
                                      suites=suites, regions=rows, region_gate=regions, paired_activation=activated,
                                      passed=activated and regions['passed'] and all(v['passed'] for v in suites.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
