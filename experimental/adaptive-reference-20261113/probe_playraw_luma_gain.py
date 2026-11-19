"""Screen target luminance as a strength cue for a fixed local chroma change."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_hdm_hdr_display import json_to
from probe_playraw_cross_editor_gain import PARENT_SHA, checked_image
from probe_playraw_mixed_light_cross_editor import FIT_SHA, REGIONS_SHA, SKIN
from probe_playraw_mixed_light_transfer import LUMA, PARENT_SHA as SOURCE_SHA, STRENGTH, chroma, read, stats
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, save
from spatial_render import finish_spatial


def fit(light, answer, weight, fixed, zone):
    sample = light[zone]
    strength = (answer[zone] @ fixed) / ((fixed @ fixed) * weight[zone])
    slope, bias = np.linalg.lstsq(np.column_stack((sample, np.ones(len(sample)))), strength, rcond=None)[0]
    return {'slope': float(slope), 'bias': float(bias), 'support': np.quantile(sample, [.01, .99]).tolist()}


def predict(model, light):
    low, high = model['support']
    supported = (light >= low) & (light <= high)
    strength = np.maximum(light * model['slope'] + model['bias'], 0)
    return np.where(supported, strength, 1.), supported


def replay(base, area, correction):
    image = base.copy()
    image[area], clipped = render(chroma(base[area]) + correction, base[area] @ LUMA)
    np.testing.assert_array_equal(base[~area], image[~area])
    return image, clipped


def accepted(score):
    return .75 <= score['projection'] <= 1.25 and score['response_mae'] <= 3


def source(root, output):
    path = root / 'playraw-mixed-light-skin-v1/replay-v2'
    identity = checked_json(path / 'results.json', SOURCE_SHA)
    fits = checked_json(root / 'playraw-mixed-light-operator-v1/results.json', FIT_SHA)['cases']
    base, _ = read(path / 'off.tif', False)
    on, mask = read(path / 'on.tif', True)
    if ([sha256(a.tobytes()).hexdigest() for a in (base, on)] != identity['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != identity['mask_pixels_sha256']):
        raise ValueError('Changed source images')
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    active, high = mask > 0, mask >= 204
    weight, light = mask.astype(float) / 255, base @ LUMA
    start = chroma(base)
    desired = chroma(finish_spatial(base, on, STRENGTH)) - start
    models, rows = [], []
    for index in (0, 1):
        fold = fits[str(index)]
        fixed = np.asarray(fold['constant'])
        model = fit(light, desired, weight, fixed, high & (parity == index))
        gain, _ = predict(model, light[active])
        result, clipped = replay(base, active, (gain * weight[active])[:, None] * fixed)
        masks = {'high': high & (parity != index), 'transition': active & ~high & (parity != index)}
        metrics = {name: stats(chroma(result) - start, desired, area) for name, area in masks.items()}
        old = fold['arms']['constant']['scores']
        passed = accepted(metrics['high']) and metrics['transition']['response_mae'] <= old['transition']['response_mae'] + .25
        rows.append({'previous': old, 'scores': metrics, 'passed': passed, 'clipped_fraction': clipped,
                     'sha256': save(output / f'source-{index}.png', result)})
        models.append(model)
    return models, rows


def transfer(parent, output, models, regions, case, fixed):
    base = checked_image(parent, case, 'baseline')
    masks = {}
    for name in SKIN:
        x0, y0, x1, y1 = regions[name]
        region = np.zeros(base.shape[:2], bool)
        region[y0:y1, x0:x1] = True
        masks[name] = region
    active = masks['cheek'] | masks['chest']
    light = base @ LUMA
    gains, supports = zip(*(predict(model, light) for model in models), strict=True)
    gain = np.mean(gains, axis=0)
    supported = np.logical_and(*supports)
    image, clipped = replay(base, active, gain[active, None] * fixed)
    digest = save(output, image)
    answer = checked_image(parent, case, 'editor-local')
    start = chroma(base)
    rows = {}
    for name, region in masks.items():
        score = stats(chroma(image) - start, chroma(answer) - start, region)
        coverage = float(supported[region].mean())
        rows[name] = {'scores': score, 'coverage': coverage, 'median_gain': float(np.median(gain[region])),
                      'luma_percentiles': np.quantile(light[region], [.01, .5, .99]).tolist(),
                      'passed': coverage >= .95 and accepted(score)}
    return {'regions': rows, 'clipped_fraction': clipped, 'sha256': digest,
            'passed': all(row['passed'] for row in rows.values())}


def run(parent, output):
    record = checked_json(parent / 'results.json', PARENT_SHA)
    regions = checked_json(Path(__file__).with_name('darktable-cuckoo-legacy-regions.json'), REGIONS_SHA)
    output.mkdir()
    models, source_rows = source(parent.parent, output)
    rows = [transfer(parent / str(i), output / f'target-{i}.png', models, regions, case,
                     np.asarray(record['fixed_chroma'])) for i, case in enumerate(record['trials'])]
    json_to(output / 'results.json', {'models': models, 'source': source_rows, 'target': rows,
                                      'passed': all(row['passed'] for row in source_rows + rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    run(**vars(parser.parse_args()))
