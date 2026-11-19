"""Use probe responses to test reference chroma orientation in ordinary pairs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_frozen_candidate_development_crosscheck import film_b
from probe_reference_black_anchor import evaluate_regions
from probe_reference_skin_chroma_isolation import skin_boxes
from probe_target_mirror_chroma import (
    CHECKPOINT, LUMA, LUMA_INTEGER, STRENGTH, SpatialEngine, background, chroma, json_to, render,
)
from spatial_render import finish_spatial


def orientation(engine, target, reference):
    y = reference @ LUMA
    color = chroma(reference, y)
    vectors = []
    for axis in np.eye(2) * 4:
        grades = []
        for direction in (-1, 1):
            guide, _ = render(color + direction * axis, y)
            grade = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
            grades.append(chroma(grade, grade @ LUMA))
        vectors.append((grades[1] - grades[0]).mean(axis=(0, 1)) / 8)
    response = np.stack(vectors)
    left, spectrum, right = np.linalg.svd(response)
    return left @ right, response, spectrum


def candidate(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {'route': 'identity'}
    if not np.any(target != target[0, 0]) or not np.any(reference != reference[0, 0]):
        return baseline.copy(), {'route': 'flat'}
    matrix, response, spectrum = orientation(engine, target, reference)
    audit = {'response': response.tolist(), 'singular_values': spectrum.tolist(), 'orientation': matrix.tolist()}
    if spectrum[-1] < .05:
        return baseline.copy(), {'route': 'weak_response', **audit}
    source = chroma(target, target @ LUMA)
    y = baseline @ LUMA
    delta = chroma(baseline, y) - source
    corrected = delta @ matrix.T
    np.testing.assert_allclose(np.linalg.norm(corrected, axis=-1), np.linalg.norm(delta, axis=-1), atol=1e-10)
    out, limited = render(source + corrected, y)
    error = np.abs((out.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000
    if error > .5:
        raise ValueError('Luma error exceeds RGB8 quantization')
    return out, {'route': 'corrected', **audit, 'gamut_limited_fraction': limited, 'max_luma_error': float(error)}


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Frozen checkpoint changed')
    flat = np.full((32, 32, 3), 100, np.uint8)
    np.testing.assert_array_equal(candidate(engine, flat, flat.copy(), 255-flat)[0], flat)
    np.testing.assert_array_equal(candidate(engine, flat, 255-flat, 255-flat)[0], 255-flat)
    out.mkdir()
    bg = background(engine, out / 'background', candidate, 'Reference response orientation')
    film = film_b(engine, out / 'film_b', candidate, 'Reference response orientation')
    checked = evaluate_regions(skin_boxes(out, bg))
    sparks = [row for name, row in film['cases'].items() if name.startswith('sparks_4592<-')]
    if len(sparks) != 2:
        raise ValueError('Missing two welder directions')
    positive = all(row['candidate'][area]['projection'] > 0 for row in sparks for area in ('whole', 'neutral'))
    style = positive and any(row['passed'] for row in sparks)
    json_to(out / 'results.json', dict(checkpoint_sha256=CHECKPOINT, background=bg, film_b=film,
                                      region_gate=checked, welder_gate=style, passed=style and bg['passed'] and checked['passed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
