"""Calibrate frozen color endpoints on already-used photos outside their training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_replication import initialize_replication_model
from audit_consistent_reference import score_outputs
from coverage_aliases import heldout_alias_closure
from independent_coco import ROOT, digest
from pairwise_gate import response_development_gate
from probe_adaptive_component_mixture import fit, input_features, pool, render
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb
from probe_training_bank_grade_anchor import MODELS, dump, forward
from spatial_render import checkpoint_for
from train_triplet_pilot import author_model


CONSUMED = {
    'reserved-expanded-v1': 'cb0ab09a3b279e49789a9c9547a0924dbcfc57dde2ed5181c8b72b7b199a9d95',
    'reserved-multiref-v1': 'e6e6e962c8993f5f8320bbc967d6f63c4d38fdcaf7a08e66e799373bd7015931',
}


def identity_set(cohort):
    return {row[role] for row in cohort for role in ('content_family', 'reference_family')}


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    audit = checked_json(base / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(data / 'cohort.json', COHORT_SHA)
    manifest = checked_json(data / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    if not (audit['passed'] and manifest['passed']) or any(manifest[k] for k in (
            'cross_split_exact_duplicates', 'cross_split_near_duplicates')):
        raise ValueError('Source isolation audit did not pass')
    entries = {item['original_path']: item for item in manifest['files']}
    training_and_dev = cohort['groups']['train'] + cohort['groups']['validation']
    known = heldout_alias_closure(cohort['alias_pairs'], identity_set(training_and_dev))
    known_pixels = {entries[r[key]]['pixel_sha256'] for r in training_and_dev for key in ('content', 'reference', 'gt')}
    calibration, assets = [], {}
    for name, sha in CONSUMED.items():
        parent = checked_json(base / name / 'results.json', sha)
        cases = parent['cases']
        if not parent['reserved_now_consumed'] or len(cases) != 32 or any('failure' in r for r in cases):
            raise ValueError('Expected previously consumed complete development evidence')
        rows = [row['triplet'] for row in cases]
        incoming = heldout_alias_closure(cohort['alias_pairs'], identity_set(rows))
        if incoming & known:
            raise ValueError('Capture families overlap')
        source = {row['path']: row for row in parent['assets']}
        hashes = {source[r[key]]['pixel_sha256'] for r in rows for key in ('content', 'reference', 'gt')}
        if known_pixels & hashes:
            raise ValueError('Exact source pixels overlap')
        known.update(incoming)
        known_pixels.update(hashes)
        calibration.extend(dict(row, source=name) for row in rows)
        assets[name] = source
    development = cohort['groups']['validation']
    rows = calibration + development
    if len(rows) != 128 or len({x['index'] for x in rows}) != 128:
        raise ValueError('Unexpected number of cases')

    def load(row, role):
        source = row.get('source')
        item = assets[source][row[role]] if source else entries[row[role]]
        folder = base / source if source else data
        return read_rgb(folder / item['prepared_path'], item['sha256'], resize=True)

    supplied = {name: checked_json(base / folder / 'scores.json', sha)
                for name, (folder, sha) in ENDPOINTS.items()}
    observations = {row['index']: (load(row, 'content'), load(row, 'reference')) for row in rows}
    model, api, _ = author_model()
    path, sha = checkpoint_for('multiref-v1')
    if digest(path) != sha:
        raise ValueError('Base model changed')
    initialize_replication_model(model, torch.load(path, map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    out.mkdir(exist_ok=False)
    endpoints, hashes = {}, {}
    for mode, (weight, identity) in MODELS.items():
        path = base / weight
        if digest(path) != identity:
            raise ValueError('Calibration endpoint changed')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        pixels = {key: forward(model, api, *pair) for key, pair in observations.items()}
        directory = base / ENDPOINTS[mode][0]
        for row in supplied[mode]['cases']:
            idx = row['index']
            image = read_rgb(directory / f'{idx}.png', row['sha256'])
            if not np.array_equal(pixels[idx], image):
                raise ValueError('One frozen development PNG did not reproduce')
        endpoint = out / f'calibration-{mode}'
        endpoint.mkdir()
        for row in calibration:
            idx = row['index']
            file = endpoint / f'{idx}.png'
            Image.fromarray(pixels[idx]).save(file)
            hashes.setdefault(idx, {})[mode] = digest(file)
        endpoints[mode] = pixels
    anchor_sha = dump(out / 'calibration-endpoints.json', hashes)

    e, c = (endpoints[mode] for mode in ('expanded', 'C10'))
    cal_gt = {row['index']: load(row, 'gt') for row in calibration}
    numeric = {}
    indexed = [row['index'] for row in calibration]
    for row in calibration:
        idx = row['index']
        numeric[idx] = dict(source=row['source'], expanded=float(np.abs(e[idx].astype(float) - cal_gt[idx]).mean()),
                            C10=float(np.abs(c[idx].astype(float) - cal_gt[idx]).mean()))

    def features(index):
        return input_features(*observations[index], e[index], c[index])

    train = np.stack([features(i) for i in indexed])
    mean, std = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-6)
    normalized = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))
    values = [pool([arm[i] for i in indexed]) for arm in (e, c, cal_gt)]
    fits = {name: fit(normalized(train), *values, flag, paired_targets=False)
            for name, flag in (('constant', False), ('adaptive', True))}
    fit_sha = dump(out / 'fit.json', dict(fitted=fits, mean=mean.tolist(), std=std.tolist()))
    produced, artifacts = {}, {}
    for name, settings in fits.items():
        folder = out / name
        folder.mkdir()
        bias, slope = (torch.tensor(settings[key]) for key in ('bias', 'slope'))
        images, decisions = {}, {}
        for row in development:
            idx = row['index']
            w = (bias + normalized(features(idx)) @ slope).sigmoid().numpy()
            output, clipped = render(*observations[idx], e[idx], c[idx], w)
            path = folder / f'{idx}.png'
            Image.fromarray(output).save(path)
            images[idx] = output
            decisions[idx] = dict(weights=w.tolist(), sha256=digest(path), clipped_channels=clipped)
        produced[name], artifacts[name] = images, decisions
    predictions_sha = dump(out / 'predictions.json', artifacts)
    truth = {row['index']: load(row, 'gt') for row in development}
    summary = {}
    for name, images in produced.items():
        metrics = score_outputs(development, images, truth)
        summary[name] = dict(mae=metrics['mean'], delta=metrics['delta_mean'],
                             gates={k: response_development_gate(supplied['current13'], metrics, supplied[k])
                                    for k in ('C10', 'control')})
    proposed = summary['adaptive']
    passed = (proposed['mae'] < min(summary['constant']['mae'], supplied['expanded']['mean'])
              and all(x['proceed_to_independent_evaluation'] for x in proposed['gates'].values()))
    elapsed = time.monotonic() - started
    if elapsed >= 120:
        raise TimeoutError('Fixed calibration budget exceeded')
    source = Path(__file__)
    result = dict(calibration=numeric, development=summary, passed=passed, elapsed=elapsed, sources=CONSUMED,
                  models={key: value[1] for key, value in MODELS.items()},
                  audit_sha256=AUDIT_SHA, cohort_sha256=COHORT_SHA, source_sha256=digest(source),
                  fit_source_sha256=digest(source.with_name('probe_adaptive_component_mixture.py')),
                  protocol_sha256=digest(source.with_name('disjoint-calibration-mixture-protocol.md')),
                  calibration_endpoints_sha256=anchor_sha, fit_sha256=fit_sha, predictions_sha256=predictions_sha)
    dump(out / 'results.json', result)
    print(json.dumps(dict(calibration_C10_wins=sum(v['C10'] < v['expanded'] - .1 for v in numeric.values()),
                          mae={k: v['mae'] for k, v in summary.items()}, passed=passed, elapsed=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
