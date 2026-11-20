"""Test bounded reference-to-edit estimation applied directly to the original target."""
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
from probe_chromamill_matched_supervision import BEYOND_SHA, CAMERAS, TAGS, TARGETS, TRAIN_SHA
from probe_chromamill_oracle_reference import PREVIOUS
from probe_chromamill_target_base import grade


CANDIDATE = 'bounded-edit-on-original'


def fit(records, excluded):
    examples = [(scene, values[kind], np.log(gain) if kind == 'digital' else np.zeros(3))
                for (scene, _), (values, gain) in records.items() if scene != excluded for kind in TAGS]
    families, raw, labels = zip(*examples, strict=True)
    families, raw, labels = np.asarray(families), np.asarray(raw), np.asarray(labels)
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), 1e-6) * np.sqrt(raw.shape[1])
    standardized = (raw - center) / scale
    distance = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2)
    cross = np.where(families[:, None] != families[None, :], distance, np.inf).min(axis=1)
    if not np.isfinite(cross).all() or np.median(cross) <= 0:
        raise ValueError('Insufficient cross-family support')
    return dict(center=center, scale=scale, support=standardized, labels=labels,
                bandwidth=float(np.median(cross)), radius=float(cross.max()))


def estimate(query, model):
    vector = (query - model['center']) / model['scale']
    distances = np.linalg.norm(model['support'] - vector, axis=1)
    nearest = float(distances.min())
    supported = nearest <= model['radius']
    if supported:
        logits = -.5 * (distances / model['bandwidth'])**2
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        log_edit = weights @ model['labels']
        if (np.any(log_edit < model['labels'].min(axis=0) - 1e-10)
                or np.any(log_edit > model['labels'].max(axis=0) + 1e-10)):
            raise ValueError('Predicted edit exceeded label component bounds')
    else:
        log_edit = np.zeros(3)
    return np.exp(log_edit), dict(nearest=nearest, supported=supported)


def audit_case(out, key, target, references, vectors, model, masks, expected, baseline):
    rendered, artifacts = {}, {}
    for kind in TAGS:
        gain, support = estimate(vectors[kind], model)
        if np.array_equal(target, references[kind]):
            result, clipped = target.copy(), 0.
        else:
            result, clipped = grade(target, gain)
        rendered[kind] = result
        artifacts[kind] = support | dict(gain=gain.tolist(), clipped_fraction=clipped,
                                        sha256=save(out / f'{key}-{kind}.png', result))
    scores = {}
    for region, mask in masks.items():
        row = compare(target, expected, *rendered.values(), mask)
        white_mae = row['style']['off_matches_original_mae']
        final_mae = row['style']['on_matches_authored_mae']
        scores[region] = dict(**row, qualified=row['both_pass'] and white_mae <= 1.5 and final_mae <= 3)
    path = out / f'{key}-comparison.png'
    panel = montage(path, [[('Fixed target', target), ('White reference', references['white']),
                            ('Real colored light', references['green']), ('Digital edit', references['digital']),
                            ('Known edit output', expected)],
                           [(f'Research 13 / {kind}', rgb) for kind, rgb in baseline.items()],
                           [(f'Bounded edit / {kind}', rgb) for kind, rgb in rendered.items()]])
    return dict(outputs=artifacts, scores=scores, panel_sha256=panel,
                training_bounds=dict(bandwidth=model['bandwidth'], radius=model['radius']))


@torch.inference_mode()
def run(prepared, data, beyond, previous, out):
    manifest = prepared / 'inputs.json'
    prior = pinned_report(previous, PREVIOUS)
    train = pinned_report(data, TRAIN_SHA)
    test = pinned_report(beyond, BEYOND_SHA)
    if digest(manifest) != test['prepared_sha256']:
        raise ValueError('Changed input manifest')
    specs = json.loads(manifest.read_bytes())['records']
    encoder = engine_for_probe()
    records, photos = {}, {}
    for scene in TARGETS:
        if scene not in train['qualified_scenes']:
            raise ValueError('Missing camera pair qualification')
        for camera in CAMERAS:
            key = scene, camera
            entry = train['scenes'][scene][camera]
            photos[key] = {kind: verified(data / f'{camera}-{scene}-{kind}.png', entry['prepared_sha256'][kind])
                           for kind in TAGS}
            vectors = {kind: features(encoder, frame)['appearance'] for kind, frame in photos[key].items()}
            records[key] = vectors, np.array(entry['linear_global_gain'])

    out.mkdir()
    chromamill, external = {}, {}
    for scene, target_key in TARGETS.items():
        model = fit(records, scene)
        target = verified(prepared / f'{target_key}.png', specs[target_key]['image_sha256'])
        masks = make_masks(target, specs[target_key]['patches'])
        for camera in CAMERAS:
            key = f'{scene}-{camera}'
            baseline_data = prior['cases'][key]
            expected = verified(previous / f'{key}-expected.png', baseline_data['expected_sha256'])
            baseline = {kind: verified(previous / f'{key}-baseline-{kind}.png',
                        baseline_data['arms']['baseline']['outputs'][f'{key}-baseline-{kind}.png']['sha256'])
                        for kind in TAGS}
            chromamill[key] = audit_case(out, key, target, photos[scene, camera], records[scene, camera][0],
                                         model, masks, expected, baseline)
    model = fit(records, None)
    for key, case in test['cases'].items():
        target = verified(prepared / f'{key}.png', specs[key]['image_sha256'])
        masks = make_masks(target, specs[key]['patches'])
        donor = case['reference']
        hashes = test['references'][donor]['sha256']
        translate = dict(zip(TAGS, ('base', 'real', 'edit'), strict=True))
        refs = {kind: verified(beyond / f'{donor}-guide-{role}.png', hashes[role])
                for kind, role in translate.items()}
        baseline = {kind: verified(beyond / f'{key}-spatial13-{role}.png',
                    case['outputs_sha256'][f'{key}-spatial13-{role}.png'])
                    for kind, role in translate.items()}
        vectors = {kind: features(encoder, frame)['appearance'] for kind, frame in refs.items()}
        expected = verified(beyond / f'{key}-expected.png', case['expected_sha256'])
        external[key] = audit_case(out, key, target, refs, vectors, model, masks, expected, baseline)
    write_json(out / 'report.json', dict(candidate=CANDIDATE, training_sha256=TRAIN_SHA, external_sha256=BEYOND_SHA,
                                       chromamill=chromamill, beyond=external))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('prepared', 'data', 'beyond', 'previous', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
