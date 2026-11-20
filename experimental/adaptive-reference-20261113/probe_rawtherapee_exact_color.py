"""Compare pooled and per-scene exact RGB chroma residual medians."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import BASELINE_SHA, load_inputs, measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_portrait_cross import managed
from probe_rawtherapee_shared_response import CUE_SHA, REGION_SHA, SKIES
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, json_to, save

RESPONSE_SHA = 'a44e2b9e59df98dd6c7ca70b987e620686e050b354184459209d6e6197f610d6'


def color_keys(before):
    red, green, blue = before.reshape(-1, 3).astype(np.int32).T
    return (red << 16) | (green << 8) | blue


def medians(keys, changes):
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    values = []
    for column in changes.T:
        ordered = column[np.lexsort((column, inverse))]
        values.append((ordered[starts + (counts - 1) // 2] + ordered[starts + counts // 2]) * .5)
    return unique, np.column_stack(values), inverse


def read_cases(root, response_dir, report):
    baseline_dir = root / 'rawtherapee-portrait-cross-v1'
    baseline = checked_json(baseline_dir / 'results.json', BASELINE_SHA)
    portraits = load_inputs(root, baseline_dir, baseline['results'])
    cue = checked_json(root / 'rawtherapee-local-edit-cue-v1/results.json', CUE_SHA)
    for name, case in report['cases'].items():
        before = (managed(root / 'rawtherapee-local-edit-cue-v1' / f'{name}-off' / 'image.png')
                  if name in SKIES else portraits[name]['off'])
        if sha256(before.tobytes()).hexdigest() != cue['inputs'][name]['rgb_sha256']['off']:
            raise ValueError(f'Changed original working RGB for {name}')
        expected = checked_rgb(response_dir / f'{name}-expected.png', case['expected_sha256'])
        prediction = checked_rgb(response_dir / f'{name}-pixel-color.png',
                                 case['arms']['pixel-color']['prediction_sha256'])
        yield name, before, expected, prediction


def means(region, supported, previous, pooled, scene, pooled_render, scene_render):
    shared = region & supported
    missing = region & ~supported
    count = int(shared.sum())
    coverage = count / int(region.sum())
    readings = {'overlap_pixels': count, 'coverage': coverage,
                'candidate_nonoverlap_mae': float(previous[missing].mean()) if missing.any() else None}
    if not count:
        return {**readings, 'scene_dependence': False}
    excess = float((pooled[shared] - scene[shared]).mean())
    return {**readings, 'candidate_overlap_mae': float(previous[shared].mean()),
            'pooled_overlap_mae': float(pooled[shared].mean()),
            'scene_overlap_mae': float(scene[shared].mean()),
            'scene_excess_mae': excess,
            'pooled_render_overlap_mae': float(pooled_render[shared].mean()),
            'scene_render_overlap_mae': float(scene_render[shared].mean()),
            'scene_dependence': count >= 128 and coverage >= .5 and excess >= 1.}


def run(root, response_dir, response_report, out):
    report = checked_json(response_report, RESPONSE_SHA)
    registered = checked_json(Path(__file__).with_name('rawtherapee-portrait-toning-regions.json'), REGION_SHA)
    cases = list(read_cases(root, response_dir, report))
    keys = [color_keys(before) for _, before, _, _ in cases]
    deltas = [(chroma(expected) - chroma(before)).reshape(-1, 2) for _, before, expected, _ in cases]
    all_keys, pooled_median, _ = medians(np.concatenate(keys), np.concatenate(deltas))
    group_counts = np.zeros(len(all_keys), dtype=int)
    for group in keys:
        group_counts[np.searchsorted(all_keys, np.unique(group))] += 1

    out.mkdir()
    regions = {}
    overlap_mae = []
    for (name, before, expected, prediction), group, wanted in zip(cases, keys, deltas):
        pooled_indices = np.searchsorted(all_keys, group)
        overlaps = group_counts[pooled_indices] > 1
        _, scene_median, scene_inverse = medians(group, wanted)
        scene_values = scene_median[scene_inverse]
        pooled_values = pooled_median[pooled_indices]
        pooled_values[~overlaps] = scene_values[~overlaps]

        original = chroma(before)
        truth = chroma(expected)
        light = before @ LUMA
        previous = np.abs(chroma(prediction) - truth).mean(-1).ravel()
        errors, render_errors, files = [], [], {}
        for label, values in (('pooled', pooled_values), ('scene', scene_values)):
            errors.append(np.abs(values - wanted).mean(-1))
            picture, _ = render(original + values.reshape(original.shape), light)
            files[label] = save(out / f'{name}-{label}.png', picture)
            render_errors.append(np.abs(chroma(picture) - truth).mean(-1).ravel())
        overlap_mae.append([int(overlaps.sum()), errors[0][overlaps].sum(), errors[1][overlaps].sum()])
        masks = measurement_masks(before, expected, registered if name == 'cuckoo' else {})
        regions[name] = {'regions': {label: means(mask.ravel(), overlaps, previous, *errors, *render_errors)
                                     for label, mask in masks.items() if mask.any()}, 'png_sha256': files}

    count, pooled_sum, scene_sum = np.sum(overlap_mae, axis=0)
    if scene_sum > pooled_sum + 1e-8:
        raise ValueError('Per-scene medians exceeded pooled optimum on overlapping keys')
    flagged = [(name, label) for name, payload in regions.items() for label, row in payload['regions'].items()
               if row['scene_dependence']]
    focus = {('cuckoo', 'whole'), ('cuckoo', 'green_background'), ('mixed', 'whole')}
    json_to(out / 'results.json', {'response_sha256': RESPONSE_SHA, 'cases': regions,
                                  'overlap_pixels': int(count),
                                  'pooled_overlap_mae': float(pooled_sum / count) if count else None,
                                  'scene_overlap_mae': float(scene_sum / count) if count else None,
                                  'scene_dependence': flagged, 'focus_scene_dependence': bool(focus.intersection(flagged))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'response_dir', 'response_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
