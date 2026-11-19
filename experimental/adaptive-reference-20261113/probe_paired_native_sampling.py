"""Evaluate native center samples against bilinear-prepared reference pairs."""

import argparse
from pathlib import Path

import numpy as np

from probe_aligned_cubic_pilot import cubic_mapping, learn, thumbnail_pair
from probe_paired_cubic_cross_content import apply, paired_fit
from probe_paired_target_weighting import bins, run


def render(original, edited, target, archived):
    table, _ = paired_fit(original, edited)
    np.testing.assert_array_equal(apply(target, table), archived)
    thumbnail, _ = thumbnail_pair(original, edited)
    height, width = thumbnail.shape[:2]
    y = (2 * np.arange(height) + 1) * original.shape[0] // (2 * height)
    x = (2 * np.arange(width) + 1) * original.shape[1] // (2 * width)
    a, b = (rgb[y[:, None], x[None, :]] for rgb in (original, edited))
    y, x = np.indices((height, width))
    train = ((x // 8 + y // 8) % 2 == 0).ravel()
    coefficient = learn(a, b, train)
    if coefficient is None:
        raise ValueError("Unstable native reference fit")
    counts = np.bincount(bins(a)[train], minlength=512)
    return apply(target, cubic_mapping(coefficient)), counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out, render, 'native')
