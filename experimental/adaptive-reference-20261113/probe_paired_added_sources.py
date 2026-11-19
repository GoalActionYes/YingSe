"""Test whether three different scenes improve paired A/B style transfer."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from probe_paired_cubic_cross_content import apply, paired_fit
from probe_paired_target_weighting import bins, run
from probe_pooled_paired_film_b import learn, prepare
from probe_target_mirror_chroma import BASE, checked_rgb
from verify_development_inputs import manifest


def reference_bank():
    suite = 'arri-two-looks-v2'
    ledger = manifest(BASE, suite)
    sources = ('poker_fullshot', 'hdr_testimage', 'sparks_4592')
    bank = {'film_a': [], 'film_b': []}
    for scene in sources:
        a, b = (ledger['preparation'][style][scene]['images'] for style in bank)
        if a['off'] != b['off']:
            raise ValueError(f"Different source frame across looks: {scene}")
        for style in bank:
            hashes = ledger['preparation'][style][scene]['images']
            pairs = [checked_rgb(BASE / suite / style / f'{scene}-{state}.png', hashes[state])
                     for state in ('off', 'on')]
            bank[style].append(prepare(pairs))
    return bank


def render(bank, original, edited, target, archived, style):
    table, _ = paired_fit(original, edited)
    np.testing.assert_array_equal(apply(target, table), archived)
    local = prepare((original, edited))
    repeated = learn([local] * 4)
    np.testing.assert_allclose(repeated.layers, table.layers, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(apply(target, repeated), archived)
    sources = [local, *bank[style]]
    added = learn(sources)
    counts = sum(np.bincount(bins(a)[train.ravel()], minlength=512) for a, _, train in sources)
    return apply(target, added), counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out, partial(render, reference_bank()), 'added')
