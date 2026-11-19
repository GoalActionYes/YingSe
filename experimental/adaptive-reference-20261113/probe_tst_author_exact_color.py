"""Compare native and resized exact RGB agreement for known authored pairs."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST
from probe_s24_exact_rgb import compare, per_half
from probe_tst_author_neighbor_replay import PARENT, PARENT_SHA
from probe_tst_paired_donor_edit import (
    SPLIT, SPLIT_SHA, checked_rgb, donor_assets, family, finish_spatial, json_to, load,
)


def readers():
    entries = {row['original_path']: row for row in load(MANIFEST, HASHES[2])['files']}

    @lru_cache(maxsize=12)
    def summarize(before, after, mode):
        images = []
        for name in (before, after):
            entry = entries[name]
            rgb = checked_rgb(COHORT.parent / entry['prepared_path'], entry['sha256'])
            if mode == 'resized':
                rgb = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))
            images.append(rgb)
        a, b = images
        return per_half(a, {'author': finish_spatial(a, b, .8)})

    return summarize


def qualifier(stats):
    if stats['colors'] < 128 or stats['target_fraction'] < .1:
        return 'insufficient_coverage'
    return 'cross_excess' if stats['cross_excess'] >= 2 else 'below_threshold'


def run(out):
    prior = load(PARENT / 'results.json', PARENT_SHA)
    sources, _ = donor_assets()
    rows = load(SPLIT / 'private-assignment.json', SPLIT_SHA)['development']
    summarize = readers()
    cases = {}
    for key, entry in prior['slots'].items():
        if 'scores' not in entry:
            continue
        index, slot = map(int, key.split('-'))
        row = rows[index]
        ref = row['refs'][slot]
        source, = sources[family(ref, 'reference')]
        truth = row['different_truth' if slot == 2 else 'same_truth']
        cases[key] = {}
        for mode in ('resized', 'native'):
            donor, _ = summarize(source, ref, mode)
            target, totals = summarize(row['content'], truth, mode)
            result = compare(donor, target, 'author', totals)
            cases[key][mode] = {label: dict(stats, status=qualifier(stats))
                                for label, stats in result.items()}
    if len(cases) != 10:
        raise ValueError('Changed prior qualification')
    out.mkdir()
    json_to(out / 'results.json', {'cases': cases, 'parent_sha256': PARENT_SHA, 'skipped': 62})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
