"""Independently allow each annotated portrait region to blend existing outputs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_gros_render_blend_bound import inequality_system, run, solve


def fit_regions(frames, target, truth, zones):
    parts = {key: mask for key, mask in zones.items() if key != 'whole'}
    coverage = np.sum(tuple(parts.values()), axis=0)
    if coverage.max() != 1 or coverage.min() != 0:
        raise ValueError('Expected non-overlapping local evaluation patches and a remainder')
    parts['remainder'] = coverage == 0
    predictions = {
        light: {state: np.empty_like(target) for state in states}
        for light, states in frames.items()
    }
    report = {}
    for name, mask in parts.items():
        rows, bounds = inequality_system(frames, target, truth, {name: mask})
        weights, lower, certificate = solve(rows, bounds)
        report[name] = {
            'weights': weights.tolist(), 'lower_bound': lower,
            'certificate': certificate, 'constraints': len(bounds),
            'pixels': int(mask.sum()),
        }
        for light, states in frames.items():
            for state, images in states.items():
                rgb = np.rint(np.tensordot(weights, images[:, mask], axes=1)).astype(np.uint8)
                predictions[light][state][mask] = rgb
    return predictions, {'regions': report}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out, render=fit_regions)
