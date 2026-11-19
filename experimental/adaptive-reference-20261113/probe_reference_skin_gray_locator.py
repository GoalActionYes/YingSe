"""Measure grayscale-only locator input on the locked background pairs."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from probe_reference_skin_chroma_attribution import CORE, OLD_SHA, enough, regions, score
from probe_reference_skin_chroma_isolation import (
    BASE, BG_SHA, CHECKPOINT, LUMA, ROOT, SpatialEngine, checked_json,
    checked_rgb, chroma, json_to, protector, render_branch, weight_for,
)
from probe_target_mirror_chroma import save


def locate_gray(model, reference):
    gray = np.rint(reference @ LUMA).astype(np.uint8)
    return weight_for(model, np.repeat(gray[..., None], 3, axis=-1))


def candidate(engine, model, target, reference, baseline, target_weight):
    if np.array_equal(target, reference):
        return target.copy(), None
    if target_weight is None:
        return baseline.copy(), None
    weight = locate_gray(model, reference)
    if weight is None:
        return baseline.copy(), None
    luma = reference @ LUMA
    frame, _, _ = render_branch(engine, target, baseline, target_weight, chroma(reference, luma) * weight[..., None], luma)
    return frame, weight


def assess_pair(engine, model, source, prior, pair, receipts, boxes, out):
    target = checked_rgb(source / 'target.png', receipts['target.png'])
    target_weight = weight_for(model, target)
    if target_weight is None:
        raise ValueError(f'Missing original target skin support for {pair}')
    out.mkdir()
    states = {}
    for state in ('off', 'on'):
        frames, old_frames, old_weights, gray_weights = [], [], [], []
        artifacts = {}
        for role in ('C1', 'F1'):
            tag = f'{role}-{state}'
            ref_name, base_name = f'{tag}-reference.png', f'{tag}-13.png'
            reference = checked_rgb(source / ref_name, receipts[ref_name])
            baseline = checked_rgb(source / base_name, receipts[base_name])
            old = prior['suites']['background']['cases'][pair]['outputs'][tag]
            old_frames.append(checked_rgb(BASE / 'reference-skin-chroma-isolation-v1' / 'background' / pair / f'{tag}.png', old['sha256']))
            weight = weight_for(model, reference)
            if weight is None or not (weight > .5).any():
                raise ValueError(f'Missing original reference skin support for {pair}/{tag}')
            frame, gray_weight = candidate(engine, model, target, reference, baseline, target_weight)
            present = gray_weight is not None
            if not present:
                gray_weight = np.zeros_like(weight)
            old_support = weight > .5
            artifacts[tag] = {'sha256': save(out / f'{tag}.png', frame), 'localized': present,
                              'retained': float((gray_weight[old_support] > .5).mean())}
            frames.append(frame)
            old_weights.append(weight)
            gray_weights.append(gray_weight)
        scores = {}
        for label, area in boxes.items():
            before = score(*old_frames, area)
            if before != prior['registered_regions'][pair][label][state]['candidate']:
                raise ValueError('Frozen region score changed')
            scores[label] = {'rgb_locator': before, 'gray_locator': score(*frames, area)}
        states[state] = {'regions': scores, 'artifacts': artifacts,
                         'rgb_weight_mae': float(np.abs(old_weights[1] - old_weights[0]).mean()),
                         'gray_weight_mae': float(np.abs(gray_weights[1] - gray_weights[0]).mean())}
    return states


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    anchor = BASE / 'bg-real-editor-look'
    frozen = checked_json(anchor / 'results.json', BG_SHA)
    prior = checked_json(BASE / 'reference-skin-chroma-isolation-v1' / 'results.json', OLD_SHA)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 weights')
    model = protector(ROOT / 'experimental/skin-protection-20261019/reference_color_app/skin.py')
    flat = np.full((64, 64, 3), 128, np.uint8)
    other = 255 - flat
    flat_weight = weight_for(model, flat)
    if flat_weight is not None or locate_gray(model, other) is not None:
        raise ValueError('Flat image should have no skin')
    np.testing.assert_array_equal(candidate(engine, model, flat, flat.copy(), other, None)[0], flat)
    np.testing.assert_array_equal(candidate(engine, model, flat, other, other, flat_weight)[0], other)
    boxes = regions()
    out.mkdir()
    cases = {pair: assess_pair(engine, model, anchor / pair, prior, pair, item['artifacts'], boxes[pair], out / pair)
             for pair, item in frozen['cases'].items()}
    primary, regressions, retention, stability = [], [], [], []
    for pair, states in cases.items():
        for state, result in states.items():
            stability.append(result['gray_weight_mae'] < result['rgb_weight_mae'])
            retention.extend(item['retained'] for item in result['artifacts'].values())
            for label, item in result['regions'].items():
                before, after = item['rgb_locator'], item['gray_locator']
                if label in CORE.get(pair, ()):
                    primary.append(enough(before, after))
                elif after - before > .25:
                    regressions.append({'pair': pair, 'state': state, 'region': label, 'increase': after - before})
    if (len(primary), len(stability), len(retention)) != (6, 8, 16):
        raise ValueError('Unexpected frozen sample counts')
    passed = sum(primary) >= 5 and not regressions and min(retention) >= .5 and sum(stability) >= 6
    json_to(out / 'results.json', {'baseline_sha256': CHECKPOINT, 'parent_sha256': OLD_SHA,
                                  'primary_improved': sum(primary), 'weight_mae_improved': sum(stability),
                                  'minimum_retention': min(retention), 'regressions': regressions,
                                  'passed': passed, 'cases': cases})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('locator_dependencies', type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    run(args.out)
