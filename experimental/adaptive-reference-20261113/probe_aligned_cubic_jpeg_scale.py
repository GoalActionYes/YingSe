"""Compare native and pooled fine residuals on JPEG-encoded reference pairs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_aligned_cubic_capture_stress import Stress
from probe_aligned_cubic_fine_residual import fine_residual, fixtures, recode
from probe_aligned_cubic_pilot import certify
from probe_arri_shared_look import metrics
from probe_frozen_render_blend_bound import neutral
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial


PREVIOUS = '91712391301411094a46619a882bcba6a1792e46cd7b474614d2b48310540aab'
ENCODINGS = ('reference_only', 'both_encoded')
ARMS = ('fine', 'pooled')


def half_mean(rgb):
    h, w = (side // 2 for side in rgb.shape[:2])
    return rgb[:2 * h, :2 * w].reshape(h, 2, w, 2, 3).mean(axis=(1, 3))


def final_scores(target, expected, output):
    common = neutral(target) & neutral(expected)
    masks = {'whole': np.ones(common.shape, dtype=bool)}
    if common.sum() >= 256:
        masks['neutral'] = common
    return {name: metrics(target, expected, target, output, mask) for name, mask in masks.items()}


def examine(folder, name, target, reference, unencoded_edit, engine):
    route, table = certify(target, reference)
    row = {'before': route['route'], 'reason_before': route.get('reason')}
    if table is None:
        return row
    estimate = apply_mapping(target, table, 1)
    cubic = finish_spatial(target, estimate, STRENGTH)
    fallback = engine.baseline(target, reference)
    decisions = {
        'fine': fine_residual(estimate, reference),
        'pooled': fine_residual(half_mean(estimate), half_mean(reference)),
    }
    expected = finish_spatial(target, unencoded_edit, STRENGTH) if unencoded_edit is not None else None
    tiles = {'target': target, 'JPEG reference': reference, '13': fallback}
    row['arms'] = {}
    for arm, decision in decisions.items():
        keep = decision['passed']
        output = cubic if keep else fallback
        summary = dict(residual=decision, route='cubic' if keep else 'legacy13',
                       sha256=save(folder / f'{name}-{arm}.png', output),
                       vs_normal=float(np.abs(output.astype(np.int16) - fallback).mean()))
        if expected is not None:
            summary['scores'] = final_scores(target, expected, output)
        row['arms'][arm] = summary
        tiles[arm] = output
    row['baseline_sha256'] = save(folder / f'{name}-13.png', fallback)
    row['panel_sha256'] = panel(folder / f'{name}-comparison.png', {
        'JPEG round trip': tiles if expected is None else {**tiles, 'expected': expected}})
    return row


def summary(rows, fixed):
    available = [name for name in fixed if 'arms' in rows['positives'][name]]
    counts = {}
    for arm in ARMS:
        negatives = [name for name, row in rows['negatives'].items()
                     if row.get('arms', {}).get(arm, {}).get('route') == 'cubic']
        positives = [name for name in available if rows['positives'][name]['arms'][arm]['route'] == 'cubic']
        passing = [name for name in positives
                   if all(value['passed'] for value in rows['positives'][name]['arms'][arm]['scores'].values())]
        counts[arm] = dict(false_positives=negatives, kept_positive=positives, passing_positive=passing,
                           passed=not negatives and len(passing) >= 6)
    return dict(available_original_positives=sorted(available),
                cubic_negatives=sum('arms' in row for row in rows['negatives'].values()), arms=counts)


@torch.inference_mode()
def run(out):
    original = BASE / 'aligned-cubic-fine-residual-v1/results.json'
    previous = checked_json(original, PREVIOUS)
    fixed = {name for name, row in previous['positives'].items() if row['route_after'] == 'cubic'}
    if len(fixed) != 8:
        raise ValueError('Changed lossless eligibility')
    out.mkdir()
    torch.set_num_threads(4)
    engine = Stress(out)
    folders = {key: out / key for key in ENCODINGS}
    for folder in folders.values():
        folder.mkdir()
    cases = {condition: {'positives': {}, 'negatives': {}} for condition in ENCODINGS}
    for group, name, target, reference in fixtures():
        compressed_ref = recode(reference)
        compressed_target = recode(target)
        positive = group == 'positives'
        if positive and certify(compressed_target, compressed_target)[0]['route'] != 'identity':
            raise ValueError('Changed exact identity route')
        for condition, base in zip(ENCODINGS, (target, compressed_target), strict=True):
            cases[condition][group][name] = examine(
                folders[condition], name, base, compressed_ref, reference if positive else None, engine)
    counts = {}
    for condition, rows in cases.items():
        if {key: len(items) for key, items in rows.items()} != {'positives': 14, 'negatives': 47}:
            raise ValueError('Changed registered pairs')
        counts[condition] = summary(rows, fixed)
    passed = all(row['arms']['pooled']['passed'] for row in counts.values())
    json_to(out / 'results.json', dict(parent_sha256=PREVIOUS, script_sha256=digest(Path(__file__)),
                                      counts=counts, cases=cases, passed=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
