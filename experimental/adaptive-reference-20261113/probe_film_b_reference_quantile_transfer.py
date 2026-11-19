"""Compare offline per-channel look deltas and the frozen spatial renderer."""
import argparse
from pathlib import Path

import numpy as np
import torch

from background_intervention import target_regions
from probe_arri_film_b_dose import endpoints
from probe_arri_shared_look import metrics
from probe_arri_two_looks import ASSIGNMENTS
from probe_target_mirror_chroma import BASE, CHECKPOINT, LUMA, STRENGTH, SpatialEngine, chroma, json_to, panel, render, save
from spatial_render import finish_spatial
from verify_development_inputs import MANIFESTS, manifest


def components(frame):
    light = frame @ LUMA
    return np.concatenate((light[..., None], chroma(frame, light)), axis=-1)


def rebase(target, reference, styled):
    if np.array_equal(reference, styled):
        return target.copy(), 0.
    signals = components(target)
    original, desired = [np.percentile(components(frame).reshape(-1, 3), np.arange(101), axis=0)
                         for frame in (reference, styled)]
    for channel in range(3):
        prior = original[:, channel]
        difference = desired[:, channel] - prior
        knots, groups, counts = np.unique(prior, return_inverse=True, return_counts=True)
        offsets = np.bincount(groups, weights=difference) / counts
        signals[..., channel] += np.interp(signals[..., channel], knots, offsets)
    return render(signals[..., 1:], np.clip(signals[..., 0], 0, 255))


def grade(engine, target, reference):
    return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)


def measure(engine, destination, frames, row):
    target, edit = frames[row['target']]
    source, edited_reference = frames[row['reference']]
    truth = finish_spatial(target, edit, STRENGTH)
    whole = np.ones(target.shape[:2], bool)
    neutral = np.logical_and(*(target_regions(x, ~whole)['neutral_proxy'] for x in (target, edit)))
    if neutral.sum() < 256:
        raise ValueError('Missing previous joint neutral support')
    ref, limited = rebase(target, source, edited_reference)
    unchanged, _ = rebase(target, source, source.copy())
    np.testing.assert_array_equal(unchanged, target)
    pairs = {'direct': (target, finish_spatial(target, ref, STRENGTH)),
             'spatial': (grade(engine, target, target), grade(engine, target, ref))}
    result, hashes, board = {}, {'guide': save(destination / 'guide.png', ref)}, {}
    for arm, (off, on) in pairs.items():
        result[arm] = {name: metrics(target, truth, off, on, mask)
                       for name, mask in (('whole', whole), ('neutral', neutral))}
        for label, picture in (('off', off), ('on', on)):
            hashes[f'{arm}-{label}'] = save(destination / f'{arm}-{label}.png', picture)
        board[arm] = {'source': target, 'guide': ref, 'off': off, 'on': on, 'known edit': truth}
    passed = {arm: all(values['passed'] for values in result[arm].values()) for arm in pairs}
    return dict(scores=result, baseline={k: row[k] for k in ('whole', 'neutral')}, passed=passed,
                gamut_limited=limited, hashes=hashes, panel_sha256=panel(destination / 'comparison.png', board))


@torch.inference_mode()
def run(out):
    source = 'arri-two-looks-v2'
    capture = manifest(BASE, source)
    historical = manifest(BASE, 'arri-film-b-crossed-v1')
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT or historical['source_sha256'] != MANIFESTS[source]:
        raise ValueError('Changed fixed research baseline')
    frames = {name: endpoints(BASE / source, capture, name) for name in ASSIGNMENTS}
    out.mkdir()
    cases = {}
    for key, row in historical['cases'].items():
        folder = out / key.replace('<-', '-from-')
        folder.mkdir()
        cases[key] = measure(engine, folder, frames, row)
    all_passed = {arm: sum(case['passed'][arm] for case in cases.values()) for arm in ('direct','spatial')}
    weld = {arm: sum(case['passed'][arm] for key, case in cases.items() if key.startswith('sparks_4592<-'))
            for arm in all_passed}
    ready = {arm: all_passed[arm] >= 4 and weld[arm] == 2 for arm in all_passed}
    json_to(out / 'results.json', dict(checkpoint_sha256=CHECKPOINT, totals=all_passed,
                                      welder_passes=weld, ready=ready, cases=cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
