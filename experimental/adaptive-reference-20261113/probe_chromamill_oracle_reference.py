"""Locate extrapolation and test a capture-aware reference replacement offline."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from probe_appearance_light_source import engine_for_probe, features
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_matched_cast import compare
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import digest, save, write_json
from probe_chromamill_matched_supervision import (
    BEYOND_SHA, CAMERAS, TAGS, TARGETS, TRAIN_SHA, fit, render,
)
from spatial_render import SpatialEngine


PREVIOUS = '73f1ead9e9c7487a49b2b8b44cbb229bb66313e557b2e15fb16ee92aeec94753'


def audit_weights(records, holdout, features_per_image, fitted):
    training = [(vectors[tag], -np.log(gain) if tag == 'green' else np.zeros(3))
                for (scene, _), (vectors, gain) in records.items() if scene != holdout for tag in TAGS]
    raw, goals = map(np.asarray, zip(*training, strict=True))
    center, scale, weights, bias = fitted
    normalize = np.sqrt(raw.shape[1])
    matrix = (raw - center) / scale / normalize
    query = (features_per_image - center) / scale / normalize
    coefficients = np.linalg.solve(matrix @ matrix.T + .1 * np.eye(len(matrix)), matrix @ query)
    coefficients += (1.0 - coefficients.sum()) / len(coefficients)
    np.testing.assert_allclose(coefficients.sum(), 1., atol=1e-12)
    raw_gain = query @ weights + bias
    np.testing.assert_allclose(coefficients @ goals, raw_gain, atol=1e-12, rtol=1e-12)
    lower, upper = goals.min(axis=0), goals.max(axis=0)
    return dict(negative_coefficient_mass=float(-coefficients[coefficients < 0].sum()),
                raw_gain=np.exp(raw_gain).tolist(), train_gain_min=np.exp(lower).tolist(),
                train_gain_max=np.exp(upper).tolist(),
                outside_training_channels=np.flatnonzero((raw_gain < lower - 1e-10) |
                                                         (raw_gain > upper + 1e-10)).tolist(),
                bounded_gain=np.exp(np.clip(raw_gain, -np.log(4), np.log(4))).tolist())


@torch.inference_mode()
def run(prepared, data, beyond, previous, out):
    prior = pinned_report(previous, PREVIOUS)
    train = pinned_report(data, TRAIN_SHA)
    target_audit = pinned_report(beyond, BEYOND_SHA)
    manifest = prepared / 'inputs.json'
    if digest(manifest) != target_audit['prepared_sha256']:
        raise ValueError('Changed prepared targets')
    specs = json.loads(manifest.read_bytes())['records']
    encoder = engine_for_probe()
    renderer = SpatialEngine('multiref-v1')
    if renderer.checkpoint_sha != prior['renderer_sha256']:
        raise ValueError('Changed target renderer')
    records, photos = {}, {}
    for scene in TARGETS:
        for camera in CAMERAS:
            key = scene, camera
            entry = train['scenes'][scene][camera]
            photos[key] = {tag: verified(data / f'{camera}-{scene}-{tag}.png', entry['prepared_sha256'][tag])
                           for tag in TAGS}
            records[key] = ({tag: features(encoder, image)['appearance'] for tag, image in photos[key].items()},
                            np.asarray(entry['linear_global_gain']))

    out.mkdir()
    cases = {}
    for scene, name in TARGETS.items():
        params = fit(records, scene, True)
        target = verified(prepared / f'{name}.png', specs[name]['image_sha256'])
        masks = make_masks(target, specs[name]['patches'])
        for camera in CAMERAS:
            tag = f'{scene}-{camera}'
            match = prior['cases'][tag]
            if match['target'] != name:
                raise ValueError('Changed fixed target')
            expected = verified(previous / f'{tag}-expected.png', match['expected_sha256'])
            reference = photos[scene, camera]
            vectors, _ = records[scene, camera]
            old, ideal, diagnostics = {}, {}, {}
            for kind in TAGS:
                key = f'{tag}-matched-{kind}.png'
                old_settings = match['arms']['matched']['outputs'][key]
                old[kind] = verified(previous / key, old_settings['sha256'])
                check = audit_weights(records, scene, vectors[kind], params)
                np.testing.assert_allclose(check['bounded_gain'], old_settings['gain'], rtol=1e-12, atol=1e-12)
                diagnostics[kind] = check
                ideal[kind] = render(renderer, target, reference['white' if kind == 'green' else kind])
                unchanged_kind = 'white' if kind == 'green' else kind
                baseline = f'{tag}-baseline-{unchanged_kind}.png'
                known = verified(previous / baseline, match['arms']['baseline']['outputs'][baseline]['sha256'])
                np.testing.assert_array_equal(ideal[kind], known)
            results = {region: compare(target, expected, *ideal.values(), mask) for region, mask in masks.items()}
            names = {kind: save(out / f'{tag}-ideal-{kind}.png', rgb) for kind, rgb in ideal.items()}
            panel = montage(out / f'{tag}-comparison.png',
                            [[('Fixed target', target), ('Known digital edit', expected)],
                             [(f'Previous / {kind}', rgb) for kind, rgb in old.items()],
                             [(f'Ideal reference / {kind}', rgb) for kind, rgb in ideal.items()]])
            cases[tag] = dict(target=name, coefficient_audit=diagnostics, ideal_scores=results,
                              outputs_sha256=names, panel_sha256=panel)
    write_json(out / 'report.json', dict(prior_sha256=PREVIOUS, scenes=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('prepared', 'data', 'beyond', 'previous', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
