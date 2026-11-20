"""Test continuous correction with matched editing negatives and scene-held outputs."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from probe_appearance_light_source import engine_for_probe, features
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_matched_cast import compare
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import STRENGTH, digest, save, write_json
from probe_physical_blue_global_fit import display, linear
from spatial_render import SpatialEngine, finish_spatial


TRAIN_SHA = '93a23c18806ccafc195baa940a71f519f49de1dc7650d66e87e698a57ad18aaf'
BEYOND_SHA = '8ba8333f5bd89afd178523459a1dda02d8029d6fb1d4bd297e289c9baaf91d6c'
CAMERAS = ('nikon', 'phone')
TAGS = ('white', 'green', 'digital')
TARGETS = {'Scene3_a': 'WHITE_green-1', 'Scene9_b': 'BLUE_red-1', 'Scene10_a': 'WHITE_green-8'}


def fit(records, holdout, hard_negative):
    raw = []
    goals = []
    tags = TAGS if hard_negative else TAGS[:2]
    for (scene, _), (probe, gain) in records.items():
        if scene == holdout:
            continue
        for label in tags:
            raw.append(probe[label])
            goals.append(-np.log(gain) if label == 'green' else np.zeros(3))
    raw, goals = np.asarray(raw), np.asarray(goals)
    center, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
    standardized = (raw - center) / scale / np.sqrt(raw.shape[1])
    bias = goals.mean(axis=0)
    dual = np.linalg.solve(standardized @ standardized.T + .1 * np.eye(len(raw)), goals - bias)
    return center, scale, standardized.T @ dual, bias


def corrected(rgb, vector, fit_parameters):
    center, scale, weights, bias = fit_parameters
    predicted_log_gain = ((vector - center) / scale) @ weights / np.sqrt(len(vector)) + bias
    log_gain = np.clip(predicted_log_gain, -np.log(4), np.log(4))
    gain = np.exp(log_gain)
    values = linear(rgb) * gain
    return display(values), dict(gain=gain.tolist(), correction_hit_gain_cap=bool(np.any(log_gain != predicted_log_gain)),
                                 reference_clipped_fraction=float(np.any(values > 1, axis=2).mean()))


def render(engine, target, reference):
    if np.array_equal(target, reference):
        return target.copy()
    return finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH)


@torch.inference_mode()
def run(prepared, data, beyond, out):
    prior = pinned_report(beyond, BEYOND_SHA)
    train = pinned_report(data, TRAIN_SHA)
    manifest = prepared / 'inputs.json'
    if not train['admission_pass'] or digest(manifest) != prior['prepared_sha256']:
        raise ValueError('Changed target or training qualification')
    specs = json.loads(manifest.read_bytes())['records']
    encoder = engine_for_probe()
    renderer = SpatialEngine('multiref-v1')
    if renderer.checkpoint_sha != prior['checkpoint_sha256']:
        raise ValueError('Changed renderer checkpoint')
    del prior
    records = {}
    pictures = {}
    for scene in TARGETS:
        if scene not in train['qualified_scenes']:
            raise ValueError('Missing qualified scene')
        for camera in CAMERAS:
            record = train['scenes'][scene][camera]
            raw = {kind: verified(data/f'{camera}-{scene}-{kind}.png', record['prepared_sha256'][kind])
                   for kind in TAGS}
            pictures[scene, camera] = raw
            records[scene, camera] = ({kind: features(encoder, image)['appearance'] for kind, image in raw.items()},
                                      np.asarray(record['linear_global_gain']))

    out.mkdir()
    cases = {}
    for scene, target_key in TARGETS.items():
        params = {label: fit(records, scene, hard) for label, hard in
                  (('physical', False), ('matched', True))}
        spec = specs[target_key]
        target = verified(prepared/f'{target_key}.png', spec['image_sha256'])
        regions = make_masks(target, spec['patches'])
        np.testing.assert_array_equal(render(renderer, target, target), target)
        for camera in CAMERAS:
            tag = f'{scene}-{camera}'
            refs = pictures[scene, camera]
            vectors, gain = records[scene, camera]
            expected = finish_spatial(target, display(linear(target) * gain), STRENGTH)
            expected_hash = save(out/f'{tag}-expected.png', expected)
            baseline = {label: render(renderer, target, reference) for label, reference in refs.items()}
            rows = [[('Target', target), ('White reference', refs['white']), ('Real green light', refs['green']),
                     ('Digital green tint', refs['digital']), ('Expected edit', expected)]]
            report = {}
            for arm in ('baseline', 'physical', 'matched'):
                details, frames = {}, {}
                for label in TAGS:
                    if arm == 'baseline':
                        result, meta = baseline[label], {}
                    else:
                        guide, meta = corrected(refs[label], vectors[label], params[arm])
                        result = render(renderer, target, guide)
                    frames[label] = result
                    key = f'{tag}-{arm}-{label}.png'
                    details[key] = dict(sha256=save(out/key, result), **meta)
                delta = np.abs(frames['white'].astype(np.int16) - baseline['white'])
                scores = {}
                for name, mask in regions.items():
                    values = compare(target, expected, *frames.values(), mask)
                    white_mae = float(delta[mask].mean())
                    scores[name] = dict(**values, white_regression_mae=white_mae,
                                        qualified=values['both_pass'] and white_mae <= 1.5)
                report[arm] = dict(outputs=details, scores=scores)
                rows.append([(f'{arm} / {label}', image) for label, image in frames.items()])
            cases[tag] = dict(target=target_key, expected_sha256=expected_hash, arms=report,
                              panel_sha256=montage(out/f'{tag}-comparison.png', rows))
    write_json(out/'report.json', dict(training_sha256=TRAIN_SHA, targets_sha256=BEYOND_SHA,
               renderer_sha256=renderer.checkpoint_sha, strength=STRENGTH, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('prepared', 'data', 'beyond', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
