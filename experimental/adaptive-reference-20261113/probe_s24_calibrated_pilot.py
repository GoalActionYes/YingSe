"""Render known Lightroom preset swaps across fixed S24 training-origin scenes."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from fetch_s24_calibrated_pilot import REVISION, ROLES
from independent_coco import digest
from probe_facd_range import dhash
from probe_reference_chroma_spread import projection
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


def image(assets, manifest, scene, role):
    path = assets / scene / (role + ROLES[role])
    item = manifest[scene][role]
    if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
        raise ValueError('S24 frozen asset changed')
    return path


def small(rgb):
    return np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))


def light_info(metadata):
    values = np.asarray([metadata[key] for key in ('gt_illum', 'pref_illum')], dtype=np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Unexpected S24 illuminant annotations')
    vectors = values / np.linalg.norm(values, axis=1, keepdims=True)
    angle = float(np.degrees(np.arccos(np.clip(vectors[0] @ vectors[1], -1, 1))))
    return dict(scene_class=metadata['scene_class'], light_class=metadata['light_class'],
                preferred_neutral_angle_degrees=angle)


def region_scores(outputs, desired, region):
    if not region.any():
        return None
    a, b, c = [value[region].astype(np.int16) for value in outputs]
    first, second = [value[region].astype(np.int16) for value in desired]
    style_delta = second - first
    alternatives = c - a, c - b
    ones = np.ones(len(style_delta), dtype=bool)
    return dict(expected_delta=float(np.abs(style_delta).mean()), same_swap=float(np.abs(a - b).mean()),
                different_swap=[float(np.abs(d).mean()) for d in alternatives],
                projection=projection((alternatives[0] + alternatives[1]) / 2, style_delta, ones),
                errors=[float(np.abs(v - gt).mean()) for v, gt in ((a, first), (b, first), (c, second))])


def run(assets, out):
    start = time.monotonic()
    torch.set_num_threads(4)
    selection = json.loads((assets / 'plan.json').read_bytes())
    manifest = json.loads((assets / 'assets.json').read_bytes())
    ids = selection['ids']
    if selection['revision'] != REVISION or len(ids) != 6 or set(ids) != manifest.keys():
        raise ValueError('Unexpected S24 case identity')
    original = {key: decode_image(image(assets, manifest, key, 'srgb_images_style_1')) for key in ids}
    fingerprints = {key: dhash(rgb) for key, rgb in original.items()}
    minimum_distance = min(int(np.count_nonzero(a != b)) for a, b in combinations(fingerprints.values(), 2))
    engine = SpatialEngine('multiref-v1')
    out.mkdir(exist_ok=False)
    predictions, assignments = {}, {}
    for index, key in enumerate(ids):
        near, far = ids[(index + 1) % 6], ids[(index + 2) % 6]
        assignments[key] = [(near, 2), (far, 2), (near, 3)]
        predictions[key] = []
        for slot, (reference, style) in enumerate(assignments[key]):
            target = original[key]
            look = decode_image(image(assets, manifest, reference, f'srgb_images_style_{style}'))
            candidate = finish_spatial(target, engine.prepare(target, look).apply(target), .8)
            path = out / f'{key}-{slot}.png'
            Image.fromarray(candidate).save(path)
            predictions[key].append(digest(path))
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n', encoding='utf-8')
    cases = {}
    for key in ids:
        target = original[key]
        authored = [decode_image(image(assets, manifest, key, f'srgb_images_style_{s}')) for s in (2, 3)]
        if any(target.shape != gt.shape for gt in authored):
            raise ValueError('S24 rendered sizes vary within a scene')
        geometry = [structure_score(small(target), small(gt)) for gt in authored]
        expected = [finish_spatial(target, gt, .8) for gt in authored]
        with Image.open(image(assets, manifest, key, 'masks')) as file:
            mask = np.asarray(file.convert('L')) > 0
        if mask.shape != target.shape[:2]:
            raise ValueError('S24 illumination mask does not align by size')
        metadata = light_info(json.loads(image(assets, manifest, key, 'data').read_bytes()))
        regions = target_regions(target, np.zeros(mask.shape, dtype=bool))
        regions = dict(whole=regions['whole'], neutral_proxy=regions['neutral_proxy'],
                       secondary_light=mask, dominant_light=~mask)
        results = [decode_image(out / f'{key}-{slot}.png') for slot in range(3)]
        measures = {label: region_scores(results, expected, selected) for label, selected in regions.items()}
        whole = measures['whole']
        passed = (whole['expected_delta'] >= 1 and whole['same_swap'] < min(whole['different_swap'])
                  and whole['projection'] is not None and whole['projection'] >= .5)
        cases[key] = dict(references=assignments[key], geometry=geometry, metadata=metadata,
                          secondary_light_pixels=int(mask.sum()), measures=measures, passed=passed)
    qualified = minimum_distance > 10 and all(g['passed'] for row in cases.values() for g in row['geometry'])
    count = sum(row['passed'] for row in cases.values())
    summary = dict(cases=cases, preds_sha256=digest(out / 'predictions.json'),
                   minimum_reference_hash_distance=minimum_distance, qualified_geometry=qualified,
                   passing_cases=count, proceed=qualified and count >= 5,
                   weight_sha256=engine.checkpoint_sha, source_sha256=digest(Path(__file__)),
                   selection_sha256=digest(assets / 'plan.json'), assets_sha256=digest(assets / 'assets.json'),
                   seconds=time.monotonic() - start)
    (out / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=count, qualified=qualified, hash_distance=minimum_distance,
                          seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.assets, args.out)
