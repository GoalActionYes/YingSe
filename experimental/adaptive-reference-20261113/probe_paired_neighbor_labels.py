"""Replace measured deltas with known deltas on the same reference RGB palette."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from probe_arri_paired_coverage import estimate, mapping
from probe_capture_invariant_look import render_style, style_transforms
from probe_paired_added_sources import reference_bank
from probe_paired_target_weighting import bins, run
from probe_pooled_paired_film_b import prepare
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb
from spatial_render import finish_spatial


PARENT = BASE / 'paired-neighbor-replay-v1'
PARENT_SHA = '253e7a696bc666d234e930fe7bce7c43b95259b9409d888be316e8a1262738a4'


def recorded_outputs():
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    frames = []
    for key, case in parent['cases'].items():
        for role, row in case['roles'].items():
            for style, digest in row['hashes'].items():
                frames.append(checked_rgb(PARENT / key / f'{role}-{style}.png', digest))
    return frames


def render(bank, processors, expected, original, edited, target, _archived, style):
    sources = [prepare((original, edited)), *bank[style]]
    before = np.concatenate([a[train] for a, _, train in sources])
    after = np.concatenate([b[train] for _, b, train in sources])
    palette, delta = mapping(before, after)
    observed, distance = estimate(palette, delta, target)
    output = finish_spatial(target, observed, STRENGTH)
    np.testing.assert_array_equal(output, next(expected))
    known = render_style(palette, processors[style]).astype(np.int16) - palette
    graded, same_distance = estimate(palette, known, target)
    np.testing.assert_array_equal(distance, same_distance)
    return finish_spatial(target, graded, STRENGTH), np.bincount(bins(before), minlength=512)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    expected = iter(recorded_outputs())
    run(parser.parse_args().out, partial(render, reference_bank(), style_transforms(BASE), expected), 'known')
    if next(expected, None) is not None:
        raise ValueError('Unused frozen control frames')
