"""Check whether tone ranks within the same material transfer across portraits."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_authored_tone_curve_replay import midrank
from probe_local_manual_luma_lock import LUMA, montage
from probe_portrait_donor_tonal_chroma import BASELINE_SHA, DIRECTIONS, render
from probe_rawtherapee_portrait_cross import CUCKOO, managed, score
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save


MIXED_CHEEK = (184, 263, 200, 280)


def regions(original, cheek):
    x0, y0, x1, y1 = cheek
    skin = np.zeros(original.shape[:2], dtype=bool)
    skin[y0:y1, x0:x1] = True
    return {'cheek': skin, 'neutral': target_regions(original, skin)['neutral_proxy'] & ~skin}


def fit(donor, edited):
    valid = ((donor >= 8) & (donor <= 247) & (edited >= 8) & (edited <= 247)).all(axis=-1)
    if not valid.any():
        raise ValueError('Empty donor range')
    brightness = donor[valid] @ LUMA
    rank = midrank(brightness)
    delta = edited[valid].astype(float) - donor[valid]
    delta -= (delta @ LUMA)[:, None]
    fixed = np.median(delta, axis=0)
    bands = np.array_split(rank.argsort(kind='stable'), 3)
    readings = [{'pixels': len(band), 'rank': float(np.median(rank[band])),
                 'brightness': float(np.median(brightness[band])), 'offset': np.median(delta[band], axis=0).tolist()}
                for band in bands if len(band)]
    centers = np.array([band['rank'] for band in readings])
    qualified = len(readings) == 3 and min(row['pixels'] for row in readings) >= 32 and np.all(np.diff(centers) > 0)
    return fixed, centers, {'bands': readings, 'fixed_offset': fixed.tolist(), 'qualified': bool(qualified)}


def predict(target, donor, source_edit, target_zones, donor_zones):
    offsets = {arm: np.zeros_like(target, dtype=float) for arm in ('constant', 'relative_rank')}
    target_brightness = target @ LUMA
    donor_rank = midrank(donor @ LUMA)
    target_rank = midrank(target_brightness)
    results = {}
    for name, target_zone in target_zones.items():
        donor_zone = donor_zones[name]
        fixed, centers, details = fit(donor[donor_zone], source_edit[donor_zone])
        target_light = target_brightness[target_zone]
        ranks = midrank(target_light)
        bands = np.array_split(ranks.argsort(kind='stable'), 3)
        details.update(donor_global_rank=float(np.median(donor_rank[donor_zone])),
                       target_global_rank=float(np.median(target_rank[target_zone])),
                       target_band_brightness=[float(np.median(target_light[band])) for band in bands])
        region_offset = fixed
        if details['qualified']:
            band_offsets = np.array([band['offset'] for band in details['bands']])
            region_offset = np.column_stack([np.interp(ranks, centers, rgb) for rgb in band_offsets.T])
        offsets['constant'][target_zone] = fixed
        offsets['relative_rank'][target_zone] = region_offset
        results[name] = details
    return results, {arm: render(target, shift) for arm, shift in offsets.items()}


def run(root, archive, out):
    prior = checked_json(archive / 'results.json', BASELINE_SHA)
    registered = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    cuckoo_path = root / 'rawtherapee-portrait-toning-v1/off/image.png'
    if digest(registered) != prior['region_sha256'] or digest(cuckoo_path) != CUCKOO['off']:
        raise ValueError('Changed portrait regions or original export')
    boxes = {'cuckoo': json.loads(registered.read_bytes())['cheek'], 'mixed': MIXED_CHEEK}
    originals = {'cuckoo': managed(cuckoo_path), 'mixed': managed(archive / 'mixed-author-off/image.png')}
    for name, pixels in originals.items():
        if sha256(pixels.tobytes()).hexdigest() != prior['results'][name]['inputs_rgb_sha256']['off']:
            raise ValueError('Changed original working RGB')
    zones = {name: regions(pixels, boxes[name]) for name, pixels in originals.items()}
    out.mkdir()
    cases, failures, gains = {}, [], []
    for name, reference in DIRECTIONS.items():
        target = originals[name]
        donor_edit = checked_rgb(archive / f'{reference}-expected.png', prior['results'][reference]['expected_sha256'])
        details, candidates = predict(target, originals[reference], donor_edit, zones[name], zones[reference])
        expected = checked_rgb(archive / f'{name}-expected.png', prior['results'][name]['expected_sha256'])
        evaluation = dict(zones[name])
        evaluation['neutral'] = evaluation['neutral'] & regions(expected, boxes[name])['neutral']
        untouched = ~(zones[name]['cheek'] | zones[name]['neutral'])
        arms = {}
        for arm, (pixels, checks) in candidates.items():
            if not np.array_equal(pixels[untouched], target[untouched]) or checks['luma_max'] > .5001:
                failures.append(f'{name}:{arm}:render')
            readings = score(target, expected, {'off': target, 'on': pixels}, evaluation)
            arms[arm] = {**checks, 'scores': readings, 'sha256': save(out / f'{name}-{arm}.png', pixels)}
        for region, mask in evaluation.items():
            original_region = target[mask]
            expected_region = expected[mask]
            delta = expected_region.astype(float) - original_region
            details[region]['target_median_offset'] = np.median(delta - (delta @ LUMA)[:, None], axis=0).tolist()
            new = arms['relative_rank']['scores'][region]
            old = arms['constant']['scores'][region]['response_mae']
            if old > 0 and new['response_mae'] <= .90 * old:
                gains.append(f'{name}:{region}')
            if not new['passed'] or not details[region]['qualified'] or new['response_mae'] > old + .25:
                failures.append(f'{name}:{region}')
        panel = montage(out / f'{name}-comparison.png', [
            [('Target original', target), ('Author edit at 80%', expected)],
            [('Region constant', candidates['constant'][0]), ('Region relative ranks', candidates['relative_rank'][0])],
        ])
        cases[name] = {'regions': details, 'arms': arms, 'untouched_pixels': int(untouched.sum()),
                       'comparison_sha256': panel}
    json_to(out / 'results.json', {'source_sha256': BASELINE_SHA, 'script_sha256': digest(Path(__file__)),
                                   'cheek_boxes': boxes, 'cases': cases, 'failures': failures,
                                   'gains': gains, 'passed': not failures and len(gains) >= 2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'archive', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
