"""Repeat the two-look benchmark with the existing RGB-neighbor operator."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from probe_arri_paired_coverage import estimate, mapping
from probe_paired_added_sources import reference_bank
from probe_paired_cubic_cross_content import apply, paired_fit
from probe_paired_target_weighting import bins, run
from probe_pooled_paired_film_b import prepare
from probe_target_mirror_chroma import STRENGTH
from spatial_render import finish_spatial


def render(bank, original, edited, target, archived, style):
    table, _ = paired_fit(original, edited)
    np.testing.assert_array_equal(apply(target, table), archived)
    sources = [prepare((original, edited)), *bank[style]]
    before = np.concatenate([a[train] for a, _, train in sources])
    after = np.concatenate([b[train] for _, b, train in sources])
    palette, delta = mapping(before, after)
    graded, _ = estimate(palette, delta, target)
    counts = np.bincount(bins(before), minlength=512)
    return finish_spatial(target, graded, STRENGTH), counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out, partial(render, reference_bank()), 'neighbors')
