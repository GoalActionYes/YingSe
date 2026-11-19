"""Test RGB-neighbor replay on the already qualified real authored pairs."""

import argparse
from pathlib import Path

import numpy as np

from probe_arri_paired_coverage import estimate, mapping
from probe_tst_paired_donor_edit import (
    BASE, RENDER_SHA, SPLIT, SPLIT_SHA, checked_rgb, donor_assets, family, finish_spatial,
    json_to, load, neutral, replay, save, score,
)


PARENT = BASE.parent / 'tst-paired-donor-edit-v1'
PARENT_SHA = 'a859decc002852860db5410b7288b9d765f99a1bcb0205a1272eb9d06edd7959'


def run(out):
    parent = load(PARENT / 'results.json', PARENT_SHA)
    rows = load(SPLIT / 'private-assignment.json', SPLIT_SHA)['development']
    source_by_family, read = donor_assets()
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    y, x = np.indices((256, 256))
    train = (x // 16 + y // 16) % 2 == 0
    cases = {}
    out.mkdir()
    for key, entry in parent['slots'].items():
        if 'scores' not in entry:
            continue
        index, slot = map(int, key.split('-'))
        row = rows[index]
        reference = row['refs'][slot]
        source, = source_by_family[family(reference, 'reference')]
        donor, edited, target = [read(path)[0] for path in (source, reference, row['content'])]
        base = checked_rgb(archive / f'{key}.png', frozen['predictions'][f'{key}.png'])
        prior = entry['scores']['donor']
        control = checked_rgb(PARENT / f'{key}-donor.png', prior['sha256'])
        np.testing.assert_array_equal(replay(donor, edited, target, train), control)
        palette, delta = mapping(donor[train], edited[train])
        graded, _ = estimate(palette, delta, target)
        frame = finish_spatial(target, graded, .8)
        authored = read(row['different_truth' if slot == 2 else 'same_truth'])[0]
        truth = finish_spatial(target, authored, .8)
        region = neutral(target) & ~train
        if int(region.sum()) != entry['neutral_pixels']:
            raise ValueError('Changed neutral coverage')
        stats = score(frame, target, truth, base, ~train, region)
        checks = score(control, target, truth, base, ~train, region)
        if any(prior[k] != v for k, v in checks.items()):
            raise ValueError('Archived control scoring changed')
        stats['sha256'] = save(out / f'{key}-neighbors.png', frame)
        cases[key] = {'control': prior, 'neighbors': stats,
                      'extra_vs_control': stats['holdout_mae'] - prior['holdout_mae']}
    skipped = {key: row for key, row in parent['slots'].items() if 'scores' not in row}
    if len(cases) != 10 or len(skipped) != 62:
        raise ValueError('Changed reference qualification')
    passed = all(row['neighbors']['passed'] and row['extra_vs_control'] <= .25 for row in cases.values())
    json_to(out / 'results.json', {'cases': cases, 'skipped': skipped, 'parent_sha256': PARENT_SHA,
                                   'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
