"""Separate the preserve/edit decision from fitting a chroma response."""

import argparse
from pathlib import Path

import numpy as np
import torch

from probe_tst_donor_supervision import (
    ASSAY, BASE, LUMA, RENDER_SHA, ROOT, SPLIT, SPLIT_SHA, SpatialEngine, checked_rgb, features,
    finish_spatial, get_color, images, json_to, load, masks, measure, neutral, render, save,
    structure_score, tensor_samples, whole,
)


ARMS = ('direct', 'gate', 'conditional')


@torch.inference_mode()
def extract(split, frozen, path):
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != frozen['checkpoint_sha256']:
        raise ValueError('Unexpected baseline weights')
    design, responses = [], []
    for index, tensors in enumerate(tensor_samples(split['training'])):
        target, refs, truths = images(tensors)
        if not all(structure_score(target, truths[i])['passed'] for i in (0, 2)):
            raise ValueError(f'Unexpected training registration: {index}')
        scope = neutral(target)
        grid = np.zeros_like(scope)
        grid[::4, ::4] = True
        scope &= grid
        for ref, truth in zip(refs, truths, strict=True):
            frame = finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
            design.append(features(target, ref, frame, scope))
            responses.append((get_color(truth) - get_color(target))[scope])
    x, y = np.concatenate(design), np.concatenate(responses)
    if len(x) != 114156:
        raise ValueError('Changed training population')
    with path.open('xb') as stream:
        np.savez_compressed(stream, x=x, y=y, split_sha256=SPLIT_SHA, render_sha256=RENDER_SHA)


def ridge(x, y, weights):
    weighted = weights[:, None] * x
    matrix = x.T @ weighted + .01 * weights.sum() * np.eye(x.shape[1])
    return np.linalg.solve(matrix, weighted.T @ y)


def fit(path):
    with np.load(path) as data:
        if str(data['split_sha256']) != SPLIT_SHA or str(data['render_sha256']) != RENDER_SHA:
            raise ValueError('Training cache identities changed')
        x, y = data['x'], data['y']
    movement = np.abs(y).max(axis=1)
    preserve, edit = movement <= 1, movement >= 3
    categories = np.where(preserve, 1 / preserve.sum(), np.where(edit, 1 / edit.sum(), 0))
    return {
        'direct': ridge(x, y, np.ones(len(x))),
        'decision': ridge(x, edit.astype(float), categories),
        'edit': ridge(x[edit], y[edit], np.ones(int(edit.sum()))),
    }, {'rows': len(x), 'edit': int(edit.sum()), 'preserve': int(preserve.sum())}


def grade(target, guide, baseline, coefficients, scope, decision_override=None):
    if np.array_equal(target, guide):
        return dict.fromkeys(ARMS, target.copy()), np.zeros_like(scope)
    x = features(target, guide, baseline, scope)
    original = get_color(target)[scope]
    base = get_color(baseline)[scope] - original
    edit = x @ coefficients['decision'] >= .5
    if decision_override is not None:
        edit = decision_override[scope]
    mask = np.zeros_like(scope)
    mask[scope] = edit
    cases = {'direct': x @ coefficients['direct'], 'gate': base * edit[:, None],
             'conditional': (x @ coefficients['edit']) * edit[:, None]}
    output = {}
    light = (baseline @ LUMA)[scope]
    cap = 2 * np.linalg.norm(base, axis=1)
    for name, delta in cases.items():
        length = np.linalg.norm(delta, axis=1)
        factor = np.divide(cap, length, out=np.ones_like(length), where=length > 0)
        frame = baseline.copy()
        frame[scope], _ = render(original + delta * np.minimum(factor, 1)[:, None], light)
        if name == 'gate':
            frame[mask] = baseline[mask]
        output[name] = frame
    return output, mask


def run(cache, out):
    torch.set_num_threads(4)
    split = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    archive = BASE / 'reference-equivalence-development-baseline-v1'
    frozen = load(archive / 'results.json', RENDER_SHA)
    assay = load(ROOT / 'docs/research/2026-11-19-tst-authored-neutral-intent-evidence.json', ASSAY)
    local = {r['index'] for r in assay['cases'] if r['aligned'] and r['one_state_protects_pixels'] >= 256}
    if len(split['training']) != 82 or len(split['development']) != 24 or len(local) != 5:
        raise ValueError('Changed cohort')
    if not cache.exists():
        extract(split, frozen, cache)
    coef, counts = fit(cache)
    out.mkdir()
    cases = {}
    for i, tensors in enumerate(tensor_samples(split['development'])):
        target, refs, truth = images(tensors)
        scope = neutral(target)
        base = [checked_rgb(archive / f'{i}-{slot}.png', frozen['predictions'][f'{i}-{slot}.png'])
                for slot in range(3)]
        attempts, decisions = zip(*(grade(target, ref, frame, coef, scope)
                                    for ref, frame in zip(refs, base, strict=True)), strict=True)
        for image in grade(target, target, base[0], coef, scope)[0].values():
            np.testing.assert_array_equal(image, target)
        authors = truth[0], truth[2]
        region = masks(target, authors) if i in local else None
        case = {'13': {'whole': whole(base, base, truth)}}
        if region is not None:
            case['13']['local'] = measure(base, authors, region)
            case['decisions'] = [
                {'false_edit_share': float(decision[region[slot == 2]].mean()) if region[slot == 2].any() else None,
                 'missed_edit_share': float((~decision[region[slot != 2]]).mean()) if region[slot != 2].any() else None}
                for slot, decision in enumerate(decisions)]
        for arm in ARMS:
            frames = [row[arm] for row in attempts]
            for frame, original in zip(frames, base, strict=True):
                np.testing.assert_array_equal(frame[~scope], original[~scope])
            case[arm] = {'whole': whole(frames, base, truth), 'hashes': [
                save(out / f'{i}-{slot}-{arm}.png', frame) for slot, frame in enumerate(frames)]}
            if region is not None:
                case[arm]['local'] = measure(frames, authors, region)
        cases[str(i)] = case
    summary = {}
    for arm in ('13', *ARMS):
        rows = [case[arm] for case in cases.values()]
        summary[arm] = {'whole': sum(r['whole']['passed'] for r in rows),
                        'local': sum(r.get('local', {}).get('passed', False) for r in rows),
                        'worst_extra_mae': max(r['whole']['worst_regression'] for r in rows)}
    final = summary['conditional']
    passed = (final['local'] >= 3 and final['local'] >= summary['direct']['local'] + 2
              and final['whole'] >= 7 and final['worst_extra_mae'] <= 2)
    json_to(out / 'results.json', {'training': counts, 'cases': cases, 'summary': summary, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
