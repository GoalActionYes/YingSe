"""Test whether local log-chroma differences identify a change of illumination."""
import argparse
from pathlib import Path

import numpy as np

from probe_beyond_rgb_edit import montage, verified
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import digest, write_json
from probe_chromamill_matched_supervision import BEYOND_SHA, CAMERAS, TAGS, TARGETS, TRAIN_SHA
from probe_physical_blue_global_fit import linear


def edges(image):
    log_rgb = np.log(np.maximum(linear(image), 1e-6))
    color = log_rgb[..., [0, 2]] - log_rgb[..., 1:2]
    valid = ((image >= 8) & (image <= 247)).all(axis=-1)
    return ((color[:, 4:] - color[:, :-4], valid[:, 4:] & valid[:, :-4]),
            (color[4:] - color[:-4], valid[4:] & valid[:-4]))


def describe(readings):
    values = np.concatenate([np.abs(delta[valid]) for delta, valid in readings])
    slots = sum(valid.size for _, valid in readings)
    if len(values) < 1024:
        raise ValueError('Insufficient supported edges')
    return np.quantile(values, [.5, .9], axis=0).ravel(), len(values) / slots


def measure(frames):
    readings = {tag: edges(image) for tag, image in frames.items()}
    descriptors = {tag: describe(result) for tag, result in readings.items()}
    shared = [np.logical_and.reduce([readings[tag][axis][1] for tag in TAGS]) for axis in range(2)]
    common = sum(int(mask.sum()) for mask in shared)
    if common < 1024:
        return descriptors, dict(edges=common, passed=False, errors={})
    errors = {}
    for tag in TAGS[1:]:
        diff = [np.abs(a[0] - b[0])[mask] for a, b, mask in zip(readings['white'], readings[tag], shared)]
        errors[tag] = float(np.median(np.concatenate(diff)))
    return descriptors, dict(edges=common, errors=errors,
                             passed=errors['green'] > 0 and errors['green'] >= 2 * errors['digital'])


def edge_preview(image):
    values = np.zeros(image.shape[:2], dtype=float)
    for position, (delta, valid) in zip((np.s_[:, 4:], np.s_[4:]), edges(image), strict=True):
        absolute = np.where(valid, np.abs(delta).mean(axis=-1), 0)
        values[position] = np.maximum(values[position], absolute)
    return np.repeat(np.rint(255 * np.clip(values / .3, 0, 1)).astype(np.uint8)[..., None], 3, axis=-1)


def classify(training, example):
    source = np.stack([values[tag][0] for values in training for tag in TAGS])
    scale = np.maximum(source.std(axis=0), 1e-6)
    centers = [source[1::3].mean(axis=0), np.concatenate((source[::3], source[2::3])).mean(axis=0)]
    margin = {}
    for tag, (values, coverage) in example.items():
        real_distance, other_distance = [float(np.linalg.norm((values - center) / scale)) for center in centers]
        margin[tag] = dict(predicted_green=real_distance < other_distance, coverage=coverage,
                           vector=values.tolist(), real_distance=real_distance, other_distance=other_distance)
    return margin


def run(camera, beyond, out):
    source = pinned_report(camera, TRAIN_SHA)
    extra = pinned_report(beyond, BEYOND_SHA)
    out.mkdir()
    data = {}
    for scene in TARGETS:
        for model in CAMERAS:
            state = source['scenes'][scene][model]['prepared_sha256']
            frames = {tag: verified(camera / f'{model}-{scene}-{tag}.png', state[tag]) for tag in TAGS}
            data[scene, model] = frames
    for scene, info in extra['references'].items():
        if not info['qualified']:
            raise ValueError('External development pair failed qualification')
        data[scene, 'beyond'] = {
            tag: verified(beyond / f'{scene}-guide-{state}.png', info['sha256'][state])
            for tag, state in zip(TAGS, ('base', 'real', 'edit'), strict=True)
        }
    measured = {key: measure(frames) for key, frames in data.items()}
    cases = {}
    totals = {group: dict(real_correct=0, false_positives=0, paired_pass=0) for group in ('camera', 'beyond')}
    for (scene, model), frames in data.items():
        group = 'beyond' if model == 'beyond' else 'camera'
        training = [values for (other, cam), (values, _) in measured.items()
                    if other != scene and cam in CAMERAS]
        values, paired = measured[scene, model]
        decisions = classify(training, values)
        total = totals[group]
        total['paired_pass'] += int(paired['passed'])
        total['real_correct'] += int(decisions['green']['predicted_green'])
        total['false_positives'] += sum(decisions[tag]['predicted_green'] for tag in ('white', 'digital'))
        name = f'{scene}-{model}'
        panel = montage(out / f'{name}.png', [
            [(tag, frames[tag]) for tag in TAGS],
            [(f'{tag} / log-chroma edges', edge_preview(frames[tag])) for tag in TAGS],
        ])
        cases[name] = dict(classification=decisions, paired=paired, panel_sha256=panel)
    passed = all(row['real_correct'] >= need and row['paired_pass'] >= need and row['false_positives'] == 0
                 for row, need in ((totals['camera'], 5), (totals['beyond'], 2)))
    write_json(out / 'report.json', dict(camera_sha256=TRAIN_SHA, beyond_sha256=BEYOND_SHA,
                                       script_sha256=digest(Path(__file__)), cases=cases, totals=totals,
                                       passed=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('camera', 'beyond', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
