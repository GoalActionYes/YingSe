"""Transfer an authored chroma change from a different photograph."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import BASELINE_SHA, DIRECTIONS, measurement_masks
from probe_local_manual_luma_lock import LUMA, locked_adjustment, montage
from probe_rawtherapee_portrait_cross import CUCKOO, managed, score
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, save


def fit(original, edited):
    valid = np.logical_and((original >= 8) & (original <= 247),
                           (edited >= 8) & (edited <= 247)).all(axis=-1)
    brightness = original[valid] @ LUMA
    if len(brightness) == 0:
        raise ValueError('No valid donor pixels')
    delta = edited[valid].astype(float) - original[valid]
    delta -= (delta @ LUMA)[:, None]
    fixed = np.median(delta, axis=0)
    bands = np.array_split(brightness.argsort(kind='stable'), 3)
    counts = [len(band) for band in bands]
    centers = np.array([np.median(brightness[band]) for band in bands]) if min(counts) >= 64 else None
    offsets = None
    if centers is not None and np.all(np.diff(centers) > 0):
        offsets = np.stack([np.median(delta[band], axis=0) for band in bands])
    return fixed, centers, offsets, counts


def render(target, offset):
    adjusted, scale = locked_adjustment(target, target + offset)
    pixels = np.rint(adjusted).astype(np.uint8)
    luma_drift = float(np.abs((pixels.astype(float) - target) @ LUMA).max())
    return pixels, {'luma_max': luma_drift, 'gamut_fraction': float((scale < 1).mean())}


def predict(target, donor, edited):
    fixed, centers, offsets, counts = fit(donor, edited)
    brightness = target @ LUMA
    outside = None
    tonal = fixed
    if offsets is not None:
        tonal = np.stack([np.interp(brightness, centers, channel) for channel in offsets.T], axis=-1)
        outside = float(((brightness < centers[0]) | (brightness > centers[-1])).mean())
    records = {'valid_fraction': sum(counts) / donor[..., 0].size, 'band_counts': counts,
               'fixed_offset': fixed.tolist(), 'centers': centers.tolist() if centers is not None else None,
               'offsets': offsets.tolist() if offsets is not None else None, 'outside_centers': outside}
    return records, {name: render(target, change) for name, change in [('fixed', fixed), ('tonal', tonal)]}


def run(root, archive, out):
    baseline = checked_json(archive / 'results.json', BASELINE_SHA)
    frozen = baseline['results']
    region_file = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    if digest(region_file) != baseline['region_sha256']:
        raise ValueError('Changed evaluation regions')
    boxes = json.loads(region_file.read_bytes())
    cuckoo = root / 'rawtherapee-portrait-toning-v1/off/image.png'
    if digest(cuckoo) != CUCKOO['off']:
        raise ValueError('Changed archived author export')
    originals = {'cuckoo': managed(cuckoo), 'mixed': managed(archive / 'mixed-author-off/image.png')}
    for name, image in originals.items():
        if sha256(image.tobytes()).hexdigest() != frozen[name]['inputs_rgb_sha256']['off']:
            raise ValueError('Changed frozen working RGB')
    out.mkdir()
    cases = {}
    for name, reference in DIRECTIONS.items():
        target, donor = originals[name], originals[reference]
        edited = checked_rgb(archive / f'{reference}-expected.png', frozen[reference]['expected_sha256'])
        donor_fit, predictions = predict(target, donor, edited)
        expected = checked_rgb(archive / f'{name}-expected.png', frozen[name]['expected_sha256'])
        masks = measurement_masks(target, expected, boxes if name == 'cuckoo' else {})
        prior = {state: checked_rgb(archive / f'{name}-from-{reference}-{state}.png', sha)
                 for state, sha in frozen[name]['predictions'].items()}
        baseline_scores = score(target, expected, prior, masks)
        if baseline_scores != frozen[name]['scores']:
            raise ValueError('Changed baseline scores')
        arms = {}
        for arm, (pixels, checks) in predictions.items():
            arms[arm] = {**checks, 'scores': score(target, expected, {'off': target, 'on': pixels}, masks),
                         'sha256': save(out / f'{name}-{arm}.png', pixels)}
        panel = montage(out / f'{name}-comparison.png', [
            [('Target original', target), ('Author edit at 80%', expected)],
            [('Donor: constant', predictions['fixed'][0]), ('Donor: three tone bands', predictions['tonal'][0])],
        ])
        cases[name] = {'fit': donor_fit, 'baseline': baseline_scores, 'arms': arms, 'comparison_sha256': panel}

    critical = ('whole', 'neutral')
    improved = []
    failures = []
    for name, case in cases.items():
        fixed, tonal = (case['arms'][arm] for arm in ('fixed', 'tonal'))
        if case['fit']['offsets'] is None or max(fixed['luma_max'], tonal['luma_max']) > .5001:
            failures.append(f'{name}:fit_or_luma')
        for region, result in tonal['scores'].items():
            old_error = fixed['scores'][region]['response_mae']
            error = result['response_mae']
            if region in critical and error <= .9 * old_error and old_error > 0:
                improved.append(f'{name}:{region}')
            if (region in critical and not result['passed']) or (region not in critical and error > old_error + .25):
                failures.append(f'{name}:{region}')
    json_to(out / 'results.json', {'input_sha256': BASELINE_SHA, 'script_sha256': digest(Path(__file__)),
                                   'cases': cases, 'improved_by_ten_percent': improved, 'failures': failures,
                                   'passed': not failures and len(improved) >= 2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'archive', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
