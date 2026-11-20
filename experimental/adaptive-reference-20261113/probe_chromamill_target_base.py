"""Compare the same inferred linear edit on the original and on the graded target base."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import STRENGTH, digest, save, write_json
from probe_chromamill_matched_supervision import BEYOND_SHA, CAMERAS, TARGETS, TRAIN_SHA
from probe_chromamill_oracle_reference import PREVIOUS
from probe_physical_blue_global_fit import display, linear
from spatial_render import finish_spatial


def fit_edit(before, after):
    a, b = linear(before), linear(after)
    yy, xx = np.indices(a.shape[:2])
    train = (yy // 16 + xx // 16) % 2 == 0
    usable = (before >= 8) & (after >= 3) & (after <= 252) & train[..., None]
    support = usable.sum(axis=(0, 1))
    if np.min(support) < 128:
        raise ValueError('Insufficient edit support per channel')
    gain = np.sum(a * b * usable, axis=(0, 1)) / np.sum(a * a * usable, axis=(0, 1))
    held = np.abs(display(a * gain).astype(np.int16) - after)[~train]
    mae, p95 = float(held.mean()), float(np.quantile(held, .95))
    return gain, dict(gain=gain.tolist(), support=support.tolist(),
                      holdout_mae=mae, holdout_p95=p95, passed=mae <= 1.5 and p95 <= 4)


def grade(base, gain):
    value = linear(base) * gain
    output = finish_spatial(base, display(value), STRENGTH)
    return output, float(np.any(value > 1, axis=2).mean())


def score(base, actual, target, expected, mask):
    response = metrics(target, expected, base, actual, mask)
    return dict(response=response, match_target_edit_mae=float(
        np.abs(actual[mask].astype(np.int16) - expected[mask]).mean()))


def run(prepared, data, beyond, previous, out):
    target_ref = pinned_report(beyond, BEYOND_SHA)
    source = pinned_report(data, TRAIN_SHA)
    prior = pinned_report(previous, PREVIOUS)
    manifest = prepared / 'inputs.json'
    if digest(manifest) != target_ref['prepared_sha256']:
        raise ValueError('Changed target files')
    records = json.loads(manifest.read_bytes())['records']
    cases = {}
    out.mkdir()
    for scene, target_key in TARGETS.items():
        target = verified(prepared / f'{target_key}.png', records[target_key]['image_sha256'])
        masks = make_masks(target, records[target_key]['patches'])
        for camera in CAMERAS:
            key = f'{scene}-{camera}'
            record = source['scenes'][scene][camera]['prepared_sha256']
            before, edited = [verified(data / f'{camera}-{scene}-{tag}.png', record[tag])
                              for tag in ('white', 'digital')]
            gain, evidence = fit_edit(before, edited)
            values = prior['cases'][key]
            expected = verified(previous / f'{key}-expected.png', values['expected_sha256'])
            baseline_keys = values['arms']['baseline']['outputs']
            base, usual = [verified(previous / f'{key}-baseline-{name}.png',
                                    baseline_keys[f'{key}-baseline-{name}.png']['sha256'])
                           for name in ('white', 'digital')]
            result = {}
            panels = [[('Reference before', before), ('Reference edit', edited),
                       ('Target', target), ('Expected edit on target', expected)]]
            for arm, input_rgb in (('original', target), ('graded_base', base)):
                output, clipped = grade(input_rgb, gain)
                scores = {region: score(input_rgb, output, target, expected, mask)
                          for region, mask in masks.items()}
                result[arm] = dict(clip=clipped,
                                   sha256=save(out / f'{key}-{arm}.png', output), scores=scores)
                panels.append([(f'{arm} / before', input_rgb), (f'{arm} / after same edit', output)])
            panels.append([('13 white reference', base), ('13 edited reference', usual)])
            cases[key] = dict(fit=evidence, paths=result, panel_sha256=montage(out / f'{key}-comparison.png', panels))
    write_json(out / 'report.json', dict(original_grade=PREVIOUS, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for arg in ('prepared', 'data', 'beyond', 'previous', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
