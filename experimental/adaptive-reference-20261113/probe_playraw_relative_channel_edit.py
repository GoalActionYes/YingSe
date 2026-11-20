"""Replay a real local edit with relative linear channel gains."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_multilum_gray_probe import apply, chroma as log_chroma, valid
from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_mixed_light_cross_editor import FIT_SHA, PROTECTED, REGIONS_SHA, SKIN
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial


def shift(base, zone, delta, weight):
    pixels = base[zone]
    changed, linear_clip = apply(pixels, delta)
    original = chroma(pixels)
    predicted = original + weight[:, None] * (chroma(changed) - original)
    light = pixels @ LUMA
    result = base.copy()
    result[zone], compressed = render(predicted, light)
    parity = dict(keep_outside=bool(np.array_equal(result[~zone], base[~zone])),
                  luma_max=float(np.abs(result[zone] @ LUMA - light).max()),
                  linear_clip=linear_clip, gamut_compressed=compressed)
    return result, parity


def bounds(score):
    return .75 <= score['projection'] <= 1.25 and score['response_mae'] <= 3


def measure(before, expected, old, new, zones):
    initial = chroma(before)
    desired = chroma(expected) - initial
    old_change = chroma(old) - initial
    change = chroma(new) - initial
    return {label: {'fixed': stats(old_change, desired, area), 'relative': stats(change, desired, area)}
            for label, area in zones.items()}


def fit_source(base, output):
    archive = base / 'playraw-mixed-light-skin-v1/replay-v2'
    info = checked_json(archive / 'results.json', SOURCE_SHA)
    fits_base = base / 'playraw-mixed-light-operator-v1'
    fits = checked_json(fits_base / 'results.json', FIT_SHA)['cases']
    before, _ = read(archive / 'off.tif', False)
    after, mask = read(archive / 'on.tif', True)
    if ([sha256(rgb.tobytes()).hexdigest() for rgb in (before, after)] != info['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != info['mask_pixels_sha256']):
        raise ValueError('Changed author pair or coverage')
    expected = finish_spatial(before, after, STRENGTH)
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    active, high = mask > 0, mask >= 204
    weight = mask.astype(float) / 255
    good = valid(before) & valid(expected)
    vectors, results = [], {}
    for index in (0, 1):
        train = high & good & (parity == index)
        count = int(train.sum())
        if count < 128:
            raise ValueError(f'Insufficient source pixels in fold {index}: {count}')
        delta = np.median((log_chroma(expected[train]) - log_chroma(before[train])) / weight[train, None], axis=0)
        vectors.append(delta)
        candidate, verification = shift(before, active, delta, weight[active])
        old = checked_rgb(fits_base / f'{index}-constant.png', fits[str(index)]['arms']['constant']['sha256'])
        held = parity != index
        zones = {'high': high & held, 'transition': active & ~high & held}
        region = measure(before, expected, old, candidate, zones)
        comparison = region['transition']
        passed = (bounds(region['high']['relative'])
                  and comparison['relative']['response_mae'] <= comparison['fixed']['response_mae'] + .25
                  and verification['keep_outside'] and verification['luma_max'] <= .5)
        results[str(index)] = dict(fit_count=count, delta=delta.tolist(), regions=region, checks=verification,
                                   passed=passed, sha=save(output / f'source-{index}.png', candidate))
    return np.mean(vectors, axis=0), results


def target_trial(folder, receipt, rectangles, delta, out):
    before, expected, old = (checked_image(folder, receipt, name)
                             for name in ('baseline', 'editor-local', 'fixed-replay'))
    zones = {}
    for label in SKIN + PROTECTED:
        x0, y0, x1, y1 = rectangles[label]
        area = np.zeros(before.shape[:2], dtype=bool)
        area[y0:y1, x0:x1] = True
        zones[label] = area
    active = zones['cheek'] | zones['chest']
    candidate, verification = shift(before, active, delta, np.ones(int(active.sum())))
    scores = measure(before, expected, old, candidate, zones)
    for name, row in scores.items():
        for key in ('projection', 'response_mae', 'response_p95'):
            if row['fixed'][key] != receipt['regions'][name][key]:
                raise ValueError(f'Changed frozen control: {name}, {key}')
        row['passed'] = (bounds(row['relative']) and receipt['regions'][name]['eligible'] if name in SKIN
                         else row['relative']['response_p95'] <= 1.5)
    return dict(regions=scores, checks=verification, sha=save(out / f'target-{folder.name}.png', candidate),
                panel_sha=panel(out / f'target-{folder.name}-comparison.png', {
                    'target': {'original': before, 'editor': expected},
                    'replay': {'fixed': old, 'relative': candidate},
                }), passed=all(row['passed'] for row in scores.values())
                and verification['keep_outside'] and verification['luma_max'] <= .5)


def run(parent, out):
    receipt = checked_json(parent / 'results.json', PARENT_SHA)
    rectangles = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    out.mkdir()
    delta, folds = fit_source(parent.parent, out)
    results = [target_trial(parent / str(i), trial, rectangles, delta, out)
               for i, trial in enumerate(receipt['trials'])]
    json_to(out / 'results.json', dict(
        sources=[SOURCE_SHA, FIT_SHA, PARENT_SHA], regions_sha=REGIONS_SHA, script_sha=digest(Path(__file__)),
        protocol_sha=digest(Path(__file__).with_name('playraw-relative-channel-edit-protocol.md')),
        source_folds=folds, mean_delta=delta.tolist(), trials=results,
        passed=all(fold['passed'] for fold in folds.values()) and all(trial['passed'] for trial in results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
