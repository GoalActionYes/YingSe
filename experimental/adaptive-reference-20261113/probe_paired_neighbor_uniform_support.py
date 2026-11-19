"""Add fixed full-gamut samples to the calibrated paired color palette."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from probe_arri_paired_coverage import estimate
from probe_capture_invariant_look import render_style, style_transforms
from probe_paired_added_sources import reference_bank
from probe_paired_target_weighting import bins, run
from probe_pooled_paired_film_b import prepare
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb
from spatial_render import finish_spatial


PARENT = BASE / 'paired-neighbor-labels-v1'
PARENT_SHA = 'bf0ed1a7dfa801e8d9298e6c5cdc0b18832c44350469f2bbd9c137ed43e5e547'


def uniform_colors():
    axis = np.rint(np.linspace(0, 255, 24)).astype(np.uint8)
    return np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)


def recorded_outputs():
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    for key, case in parent['cases'].items():
        for role, row in case['roles'].items():
            for style, digest in row['hashes'].items():
                yield checked_rgb(PARENT / key / f'{role}-{style}.png', digest)


def render(bank, processors, uniform, expected, original, edited, target, _archived, style):
    sources = [prepare((original, edited)), *bank[style]]
    before = np.concatenate([a[train] for a, _, train in sources])
    palette = np.unique(before, axis=0)
    known = render_style(palette, processors[style]).astype(np.int16) - palette
    original_estimate, _ = estimate(palette, known, target)
    np.testing.assert_array_equal(finish_spatial(target, original_estimate, STRENGTH), next(expected))
    augmented = np.unique(np.concatenate((palette, uniform)), axis=0)
    delta = render_style(augmented, processors[style]).astype(np.int16) - augmented
    graded, _ = estimate(augmented, delta, target)
    return finish_spatial(target, graded, STRENGTH), np.bincount(bins(augmented), minlength=512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    expected = recorded_outputs()
    renderer = partial(render, reference_bank(), style_transforms(BASE), uniform_colors(), expected)
    run(parser.parse_args().out, renderer, 'uniform')
    if next(expected, None) is not None:
        raise ValueError('Unused frozen control frames')
