"""Bound what reducing a fixed cross-photo chroma shift can repair offline."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from probe_hdm_hdr_display import digest, json_to
from probe_playraw_mixed_light_cross_editor import FIT_SHA, PROTECTED, REGIONS_SHA, SKIN
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, save
from spatial_render import finish_spatial


PARENT_SHA = '030d3a3fabc5d50bff888d415f7095071bb0e364de9501263d5b96b494e06ca7'


def checked_image(folder, row, name):
    path = folder / f'{name}.png'
    if digest(path) != row['images'][name]:
        raise ValueError(f'Changed parent image: {path}')
    with Image.open(path) as image:
        return np.array(image)


def gains(desired, fixed):
    # With two nonzero components, either the heavier breakpoint or any tied interval minimizes L1.
    weight = np.abs(fixed)
    if np.any(weight == 0):
        raise ValueError('Expected two nonzero source components')
    ratios = desired / fixed
    breaks = ratios.ravel()
    order = np.argsort(breaks)
    cumulative = np.cumsum(np.broadcast_to(weight, ratios.shape).ravel()[order])
    median = breaks[order[np.searchsorted(cumulative, cumulative[-1] / 2)]]
    per_pixel = np.maximum(ratios[:, np.argmax(weight)], 0)
    return float(np.clip(median, 0, 1)), per_pixel


def trial(parent_path, parent_row, rectangles, fixed, output):
    base, editor, old = [checked_image(parent_path, parent_row, name)
                         for name in ('baseline', 'editor-local', 'fixed-replay')]
    regions = {}
    for name in SKIN + PROTECTED:
        x0, y0, x1, y1 = rectangles[name]
        region = np.zeros(base.shape[:2], dtype=bool)
        region[y0:y1, x0:x1] = True
        regions[name] = region
    skin = regions['cheek'] | regions['chest']
    start = chroma(base)
    desired, old_change = chroma(editor) - start, chroma(old) - start
    gain, per_pixel = gains(desired[skin], fixed)
    candidate = base.copy()
    candidate[skin], clipped = render(start[skin] + gain * fixed, base[skin] @ LUMA)
    np.testing.assert_array_equal(candidate[~skin], base[~skin])
    bound = np.zeros(base.shape[:2])
    bound[skin] = np.abs(desired[skin] - per_pixel[:, None] * fixed).mean(axis=1)
    change = chroma(candidate) - start
    results = {}
    for name, region in regions.items():
        baseline, score = [stats(actual, desired, region) for actual in (old_change, change)]
        for key in ('projection', 'response_mae', 'response_p95'):
            expected = parent_row['regions'][name][key]
            if baseline[key] is None:
                if expected is not None:
                    raise ValueError(f'Changed baseline projection: {name}')
            else:
                np.testing.assert_allclose(baseline[key], expected)
        in_skin = name in SKIN
        passed = (.75 <= score['projection'] <= 1.25 and score['response_mae'] <= 3
                  if in_skin else score['response_p95'] <= 1.5)
        results[name] = {'old': baseline, 'reduced': score, 'unquantized_pixel_bound': float(bound[region].mean()),
                         'passed': passed}
    return {'gain': gain, 'regions': results, 'clipped_fraction': clipped, 'image_sha256': save(output, candidate),
            'passed': all(row['passed'] for row in results.values())}


def source_regression(root, gain, out):
    source = root / 'playraw-mixed-light-skin-v1/replay-v2'
    identity = checked_json(source / 'results.json', SOURCE_SHA)
    folds = checked_json(root / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    base, _ = read(source / 'off.tif', False)
    edited, mask = read(source / 'on.tif', True)
    if ([sha256(pixels.tobytes()).hexdigest() for pixels in (base, edited)] != identity['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != identity['mask_pixels_sha256']):
        raise ValueError('Changed source photo pair')
    base_chroma = chroma(base)
    desired = chroma(finish_spatial(base, edited, STRENGTH)) - base_chroma
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    active, high = mask > 0, mask >= 204
    weight = mask[active].astype(float) / 255
    results = {}
    for index in (0, 1):
        fold = folds[str(index)]
        shifted = base.copy()
        delta = np.asarray(fold['constant']) * gain
        shifted[active], _ = render(base_chroma[active] + weight[:, None] * delta, base[active] @ LUMA)
        np.testing.assert_array_equal(shifted[~active], base[~active])
        change = chroma(shifted) - base_chroma
        held = parity != index
        zones = {'high': high & held, 'transition': active & ~high & held}
        scores = {name: stats(change, desired, region) for name, region in zones.items()}
        old = fold['arms']['constant']['scores']
        passed = (.75 <= scores['high']['projection'] <= 1.25 and scores['high']['response_mae'] <= 3
                  and scores['transition']['response_mae'] - old['transition']['response_mae'] <= .25)
        results[str(index)] = {'old': old, 'reduced': scores, 'passed': passed,
                               'image_sha256': save(out / f'source-{index}-reduced.png', shifted)}
    return results


def run(parent, out):
    evidence = checked_json(parent / 'results.json', PARENT_SHA)
    rectangles = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    fixed = np.asarray(evidence['fixed_chroma'])
    out.mkdir()
    trials = [trial(parent / str(i), record, rectangles, fixed, out / f'{i}-reduced.png')
              for i, record in enumerate(evidence['trials'])]
    np.testing.assert_equal(trials[0]['gain'], trials[1]['gain'])
    regression = source_regression(parent.parent, trials[0]['gain'], out)
    json_to(out / 'results.json', {'trials': trials, 'parent_sha256': PARENT_SHA,
                                   'source_regression': regression,
                                   'parent_full_frame_repeatable': evidence['repeatable']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
