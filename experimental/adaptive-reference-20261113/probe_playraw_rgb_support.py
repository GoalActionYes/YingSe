"""Measure two-fold RGB support for cross-photo replay of one known local edit."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_hdm_hdr_display import json_to
from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_luma_gain import accepted, replay
from probe_playraw_mixed_light_cross_editor import REGIONS_SHA, SKIN
from probe_playraw_mixed_light_transfer import PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_target_mirror_chroma import checked_json, save
from spatial_render import finish_spatial


def donors(root):
    folder = root / 'playraw-mixed-light-skin-v1/replay-v2'
    record = checked_json(folder / 'results.json', SOURCE_SHA)
    base, _ = read(folder / 'off.tif', False)
    on, weight = read(folder / 'on.tif', True)
    if ([sha256(a.tobytes()).hexdigest() for a in (base, on)] != record['rgb_sha256']
            or sha256(weight.tobytes()).hexdigest() != record['mask_pixels_sha256']):
        raise ValueError('Changed source evidence')
    authored = chroma(finish_spatial(base, on, STRENGTH)) - chroma(base)
    y, x = np.indices(weight.shape)
    parity = (x // 8 + y // 8) % 2
    groups = []
    for i in (0, 1):
        selected = (parity == i) & (weight >= 204)
        groups.append((base[selected].astype(np.int16), authored[selected] * 255 / weight[selected, None]))
    return groups


def nearest(rgb, group):
    colors, changes = group
    indices, distances = [], []
    for start in range(0, len(rgb), 128):
        distance = np.abs(rgb[start:start+128, None] - colors).max(axis=-1)
        index = np.argmin(distance, axis=1)
        indices.extend(index)
        distances.extend(distance[np.arange(len(index)), index])
    return np.asarray(distances), changes[indices]


def trial(folder, case, rectangles, groups, fixed, out):
    base = checked_image(folder, case, 'baseline')
    masks = {}
    for name in SKIN:
        x0, y0, x1, y1 = rectangles[name]
        area = np.zeros(base.shape[:2], dtype=bool)
        area[y0:y1, x0:x1] = True
        masks[name] = area
    chosen = masks['cheek'] | masks['chest']
    distance, response = zip(*(nearest(base[chosen].astype(np.int16), group) for group in groups), strict=True)
    worst_distance = np.maximum(*distance)
    supported = worst_distance <= 2
    shift = np.where(supported[:, None], np.mean(response, axis=0), fixed)
    result, clipped = replay(base, chosen, shift)
    digest = save(out, result)
    target = checked_image(folder, case, 'editor-local')
    old = checked_image(folder, case, 'fixed-replay')
    start = chroma(base)[chosen]
    expected, actual, previous = (chroma(img[chosen]) - start for img in (target, result, old))
    disagreement = np.abs(response[0] - response[1]).mean(axis=1)
    rows = {}
    for name, mask in masks.items():
        region = mask[chosen]
        subset = region & supported
        count = int(subset.sum())
        score = stats(actual, expected, region)
        coverage = {str(level): float((worst_distance[region] <= level).mean()) for level in (0, 2, 5)}
        rows[name] = {'supported_pixels': count, 'coverage': coverage,
                      'distance_p50_p95': np.quantile(worst_distance[region], [.5, .95]).tolist(),
                      'full': score, 'old': stats(previous, expected, region),
                      'supported': stats(actual, expected, subset) if count else None,
                      'old_supported': stats(previous, expected, subset) if count else None,
                      'source_fold_disagreement': float(disagreement[subset].mean()) if count else None,
                      'passed': count >= 128 and coverage['2'] >= .8 and accepted(score)}
    return {'rows': rows, 'sha256': digest, 'clipped_fraction': clipped,
            'passed': all(record['passed'] for record in rows.values())}


def run(parent, out):
    record = checked_json(parent / 'results.json', PARENT_SHA)
    regions = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    groups = donors(parent.parent)
    out.mkdir()
    pairs = [trial(parent / str(i), case, regions, groups, np.asarray(record['fixed_chroma']), out / f'{i}-rgb.png')
             for i, case in enumerate(record['trials'])]
    json_to(out / 'results.json', {'trials': pairs, 'source_counts': [len(group[0]) for group in groups],
                                   'parent_full_frame_repeatable': record['repeatable'],
                                   'passed': all(row['passed'] for row in pairs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
