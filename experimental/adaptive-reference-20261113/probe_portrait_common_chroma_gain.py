"""Determine whether one idealized color-strength setting can recover a portrait edit."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import BASELINE_SHA, DIRECTIONS, load_inputs, measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_portrait_reference_paths import EXPANDED_SHA
from probe_rawtherapee_portrait_cross import score
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save


def range_for(projection):
    if projection is None or projection <= 0:
        return None
    return [.75 / projection, 1.25 / projection]


def intersection(ranges):
    if any(row is None for row in ranges):
        return None
    low = max(row[0] for row in ranges)
    high = min(row[1] for row in ranges)
    return [low, high] if low <= high else None


def evaluate(target, expected, archive, baseline, masks, output):
    if score(target, expected, archive, masks) != baseline:
        raise ValueError('Archived regional scores changed')
    ranges = {name: range_for(row['projection']) for name, row in baseline.items() if row['qualified']}
    critical = [ranges[name] for name in ('whole', 'neutral')]
    ideal = chroma(expected) - chroma(target)
    observed = chroma(archive['on']) - chroma(archive['off'])
    squared = np.square(observed).sum()
    gain = float(max(0, (ideal * observed).sum() / squared)) if squared else 0.
    pixels, clipped = render(chroma(target) + gain * observed, target @ LUMA)
    scores = score(target, expected, {'off': target, 'on': pixels}, masks)
    return {'ranges': ranges, 'required_intersection': intersection(critical),
            'all_registered_intersection': intersection(list(ranges.values())),
            'gain': gain, 'gamut_fraction': clipped, 'rendered_scores': scores, 'sha256': save(output, pixels)}


def run(root, original, expanded, out):
    base = checked_json(original / 'results.json', BASELINE_SHA)
    extra = checked_json(expanded / 'results.json', EXPANDED_SHA)
    region_file = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    if digest(region_file) != base['region_sha256']:
        raise ValueError('Changed measurement regions')
    boxes = json.loads(region_file.read_bytes())
    images = load_inputs(root, original, base['results'])
    out.mkdir()
    rows = {}
    for name, reference in DIRECTIONS.items():
        item = base['results'][name]
        target = images[name]['off']
        expected = checked_rgb(original / f'{name}-expected.png', item['expected_sha256'])
        regions = measurement_masks(target, expected, boxes if name == 'cuckoo' else {})
        main_archive = {state: checked_rgb(original / f'{name}-from-{reference}-{state}.png', value)
                        for state, value in item['predictions'].items()}
        expanded_item = extra['results'][name]
        added_archive = {state: checked_rgb(expanded / f'{name}-expanded-{state}.png', value)
                         for state, value in expanded_item['outputs']['expanded'].items()}
        bases = {'13': (main_archive, item['scores']),
                 'expanded': (added_archive, expanded_item['scores']['expanded'])}
        rows[name] = {arm: evaluate(target, expected, saved, scores, regions, out / f'{name}-{arm}.png')
                      for arm, (saved, scores) in bases.items()}
    json_to(out / 'results.json', {'sources': [BASELINE_SHA, EXPANDED_SHA], 'cases': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'original', 'expanded', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
