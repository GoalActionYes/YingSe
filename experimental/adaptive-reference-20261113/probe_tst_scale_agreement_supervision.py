"""Select trusted paired supervision by comparing native and resized labels."""

import argparse
from pathlib import Path

import numpy as np

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST
from probe_tst_author_native_fit import grid_centers
from probe_tst_donor_supervision import ARMS, fit, run
from probe_tst_neutral_direction_fit import get_color
from probe_tst_paired_donor_edit import checked_rgb, load, replay


def fit_with_agreement(split, frozen):
    entries = {row['original_path']: row for row in load(MANIFEST, HASHES[2])['files']}
    agreed_counts = []

    def extra_labels(source, guide, target, train, area, author_delta, teacher_delta):
        before, after = [checked_rgb(COHORT.parent / entries[path]['prepared_path'], entries[path]['sha256'])
                         for path in (source, guide)]
        if before.shape != after.shape:
            raise ValueError('Native donor pair dimensions disagree')
        y, x = (grid_centers(length) for length in before.shape[:2])
        a, b = (image[y[:, None], x[None, :]][train] for image in (before, after))
        graded = replay(a, b, target, np.ones(len(a), dtype=bool))
        native_delta = get_color(graded)[area] - get_color(target)[area]
        agree = np.linalg.norm(native_delta - teacher_delta, axis=1) <= 1
        labels = author_delta.copy()
        labels[agree] = .5 * author_delta[agree] + .25 * (teacher_delta[agree] + native_delta[agree])
        agreed_counts.append(int(agree.sum()))
        return labels

    coefficients, training = fit(split, frozen, extra_labels=extra_labels)
    if (len(agreed_counts), training['rows'], training['groups']) != (137, 57253, 71):
        raise ValueError('Original paired supervision scope changed')
    return coefficients, {**training, 'agreed_rows': sum(agreed_counts), 'agreed_per_slot': agreed_counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out, fit_with_agreement, (*ARMS, 'agreement'))
