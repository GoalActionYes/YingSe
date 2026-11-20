"""Hold reference RGB distribution constant while reversing spatial layout."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_target_mirror_chroma import (
    BG_SHA, CHECKPOINT, FACD_SHA, LUMA, LUMA_INTEGER, STRENGTH, SpatialEngine, background,
    chroma, digest, facd, finish_spatial, json_to, render,
)


def grade(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'identity': True, 'max_luma_error': 0.0}
    reverse = np.ascontiguousarray(reference[:, ::-1])
    opposite = finish_spatial(target, engine.prepare(target, reverse).apply(target), STRENGTH)
    baseline_y, opposite_y = baseline @ LUMA, opposite @ LUMA
    a, b = chroma(baseline, baseline_y), chroma(opposite, opposite_y)
    result, compressed = render((a + b) / 2, baseline_y)
    return result, {
        'reference_mirror_output_chroma_mae': float(np.abs(a - b).mean()),
        'max_luma_error': float(np.abs((result.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000),
        'gamut_fraction': compressed,
        'output_change_mae': float(np.abs(result.astype(np.int16) - baseline).mean()),
    }


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed checkpoint')
    toy = np.array([[[0, 10, 100], [128, 129, 240]]], dtype=np.uint8)
    same, _ = grade(engine, toy, toy.copy(), 255 - toy)
    np.testing.assert_array_equal(same, toy)
    out.mkdir()
    bg = background(engine, out / 'bg-real', grade=grade, label='reference mirror')
    author = facd(engine, out / 'facd', grade=grade, label='reference mirror')
    diagnostics = [row for block in (bg, author) for case in block['cases'].values()
                   for row in case['outputs'].values()]
    max_error = max(row['max_luma_error'] for row in diagnostics)
    json_to(out / 'results.json', dict(
        checkpoint=CHECKPOINT, strength=STRENGTH, sources=[BG_SHA, FACD_SHA],
        script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
            'reference-mirror-chroma-protocol.md')), background=bg, facd=author, max_luma_error=max_error,
        max_reference_mirror_chroma_mae=max(row['reference_mirror_output_chroma_mae'] for row in diagnostics),
        passed=bg['passed'] and author['passed'] and max_error <= .5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
