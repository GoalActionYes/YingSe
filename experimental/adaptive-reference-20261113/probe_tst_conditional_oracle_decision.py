"""Replay learned edit vectors with offline author preserve/edit decisions."""

import argparse
from hashlib import file_digest
from pathlib import Path

import numpy as np

from probe_tst_conditional_chroma import ARMS, fit, grade
from probe_tst_donor_supervision import (
    BASE, RENDER_SHA, SPLIT, SPLIT_SHA, checked_rgb, images,
    json_to, load, masks, measure, neutral, save, tensor_samples,
)


PARENT = BASE.parent / 'tst-conditional-chroma-v1'
PARENT_SHA = '600f2f640e7b87848c23f8b8b5839c30db474b3d62ef21312c321fcf8330fc86'
CACHE_SHA = 'f483c31f021b8ea2337806800cb080142e28893c2ec9b6b0a9cfa681de43f353'


def run(cache, out):
    with cache.open('rb') as stream:
        if file_digest(stream, 'sha256').hexdigest() != CACHE_SHA:
            raise ValueError('Changed frozen training cache')
    coefficients, counts = fit(cache)
    parent = load(PARENT / 'results.json', PARENT_SHA)
    if counts != parent['training']:
        raise ValueError('Changed training counts')
    split = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    out.mkdir()
    results = {}
    reproduced = 0
    for index, tensors in enumerate(tensor_samples(split['development'])):
        target, refs, truth = images(tensors)
        prior = parent['cases'][str(index)]
        scope = neutral(target)
        base = [checked_rgb(archive / f'{index}-{slot}.png', frozen['predictions'][f'{index}-{slot}.png'])
                for slot in range(3)]
        existing = []
        for slot, (reference, frame) in enumerate(zip(refs, base, strict=True)):
            output, decision = grade(target, reference, frame, coefficients, scope)
            for arm in ARMS:
                path = PARENT / f'{index}-{slot}-{arm}.png'
                np.testing.assert_array_equal(output[arm], checked_rgb(path, prior[arm]['hashes'][slot]))
                reproduced += 1
            existing.append((output, decision))
        if 'local' not in prior['conditional']:
            continue
        authors = (truth[0], truth[2])
        protect = masks(target, authors)
        selected = protect[0] | protect[1]
        corrected = {'gate': [], 'conditional': []}
        for slot, (guide, frame, (outputs, decision)) in enumerate(zip(refs, base, existing, strict=True)):
            forced = decision.copy()
            forced[protect[slot == 2]] = False
            forced[protect[slot != 2]] = True
            attempt, _ = grade(target, guide, frame, coefficients, scope, decision_override=forced)
            for arm in corrected:
                np.testing.assert_array_equal(attempt[arm][~selected], outputs[arm][~selected])
                corrected[arm].append(attempt[arm])
        results[str(index)] = {}
        for arm, frames in corrected.items():
            results[str(index)][arm] = {
                'old': prior[arm]['local'], 'oracle': measure(frames, authors, protect),
                'hashes': [save(out / f'{index}-{slot}-{arm}.png', frame)
                           for slot, frame in enumerate(frames)],
            }
    if reproduced != 216 or len(results) != 5:
        raise ValueError('Frozen inputs or qualification changed')
    totals = {arm: sum(row[arm]['oracle']['passed'] for row in results.values())
              for arm in ('gate', 'conditional')}
    passed = totals['conditional'] >= 3 and totals['conditional'] >= totals['gate'] + 2
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'cache_sha256': CACHE_SHA,
                                   'cases': results, 'totals': totals, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
