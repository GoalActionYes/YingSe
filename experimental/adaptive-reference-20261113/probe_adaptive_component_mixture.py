"""Fit independent image-level luminance and chroma mixing from paired training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_replication import initialize_replication_model
from audit_consistent_reference import score_outputs
from background_intervention import LUMA
from independent_coco import ROOT, digest
from pairwise_gate import response_development_gate
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb
from probe_training_bank_grade_anchor import MODELS, dump, forward
from spatial_render import checkpoint_for
from train_triplet_pilot import author_model


def coordinates(rgb):
    x = rgb.astype(np.float64) / 255
    y = x @ LUMA
    return np.stack((y, x[..., 2] - y, x[..., 0] - y), axis=-1)


def input_features(target, reference, expanded, control):
    a, b = coordinates(target), coordinates(reference)
    signed = (control.astype(np.float64) - expanded) / 255
    light = signed @ LUMA
    chroma = signed - light[..., None]
    return np.r_[(b.mean((0, 1)) - a.mean((0, 1))), (b.std((0, 1)) - a.std((0, 1))),
                 np.abs(light).mean(), np.abs(chroma).mean()]


def train_cases(rows, holdout):
    groups = defaultdict(list)
    for r in rows:
        groups[r['group_id']].append(r)
    ids = [key for key in sorted(groups, key=int) if len(groups[key]) == 4][:64]
    selected = [row for key in ids for row in sorted(groups[key], key=lambda r: r['style_index'])]
    if len(selected) != 256 or len({r['index'] for r in selected}) != 256:
        raise ValueError('Incomplete training subset')
    family = lambda cohort: {r[k] for r in cohort for k in ('content_family', 'reference_family')}
    if family(selected) & family(holdout):
        raise ValueError('Training and development families overlap')
    return selected


def pool(rows):
    shape = (-1, 32, 8, 32, 8, 3)
    return torch.from_numpy(np.stack(rows).reshape(shape).mean(axis=(2, 4)).astype(np.float32) / 255)


def fit(features, expanded, C10, truth, active, *, paired_targets=True):
    difference = C10 - expanded
    light = (difference * torch.tensor(LUMA, dtype=torch.float32)).sum(-1, keepdim=True)
    residual = difference - light
    bias = torch.full((2,), float(np.log(1. / 9)), requires_grad=True)
    slope = torch.zeros((features.shape[1], 2), requires_grad=True)
    optimizer = torch.optim.Adam([bias, slope] if active else [bias], lr=.04)
    if paired_targets:
        train_delta = truth.reshape(64, 4, 32, 32, 3)
        train_delta = train_delta[:, 1:] - train_delta[:, :1]
    losses = []
    for step in range(180):
        optimizer.zero_grad()
        logits = bias + (features @ slope if active else torch.zeros((features.shape[0], 2)))
        weights = logits.sigmoid()[:, None, None, :]
        prediction = (expanded + weights[..., :1] * light + weights[..., 1:] * residual).clamp(0, 1)
        rgb = (prediction - truth).abs().mean()
        if paired_targets:
            stack = prediction.reshape(64, 4, 32, 32, 3)
            paired = (stack[:, 1:] - stack[:, :1] - train_delta).abs().mean()
        else:
            paired = rgb.new_zeros(())
        loss = rgb + .5 * paired + (.002 * slope.square().sum() if active else 0)
        if step == 0 or step == 179:
            losses.append([rgb.item(), paired.item(), loss.item()])
        loss.backward()
        optimizer.step()
    if not torch.isfinite(slope).all() or not torch.isfinite(bias).all():
        raise ValueError('Nonfinite mixture fit')
    return dict(bias=bias.detach().tolist(), slope=slope.detach().tolist(), training_losses=losses)


def render(target, reference, expanded, C10, weights):
    if np.array_equal(target, reference):
        return target.copy(), 0
    delta = C10.astype(np.float64) - expanded
    light = (delta @ LUMA)[..., None]
    raw = expanded + weights[0] * light + weights[1] * (delta - light)
    rounded = np.rint(raw)
    clipped = int(np.count_nonzero((rounded < 0) | (rounded > 255)))
    return rounded.clip(0, 255).astype(np.uint8), clipped


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
    if not audit['passed'] or not manifest['passed']:
        raise ValueError('Frozen data audit did not pass')
    entries = {r['original_path']: r for r in manifest['files']}
    development = cohort['groups']['validation']
    train = train_cases(cohort['groups']['train'], development)
    rows = train + development
    if len(rows) != 320 or len({r['index'] for r in rows}) != 320:
        raise ValueError('Unexpected group identity')

    def image(path):
        entry = entries[path]
        return read_rgb(data / entry['prepared_path'], entry['sha256'], resize=True)

    provided = {}
    for name, (dir_name, sha) in ENDPOINTS.items():
        provided[name] = checked_json(base / dir_name / 'scores.json', sha)
    actual_inputs = {r['index']: (image(r['content']), image(r['reference'])) for r in rows}
    model, api, _ = author_model()
    path, sha = checkpoint_for('multiref-v1')
    if digest(path) != sha:
        raise ValueError('Initial identity changed')
    initialize_replication_model(model, torch.load(path, weights_only=True, map_location='cpu'))
    model.eval().requires_grad_(False)
    endpoint_images = {}
    for name, (file, expected) in MODELS.items():
        path = base / file
        if digest(path) != expected:
            raise ValueError('Frozen calibration endpoint changed')
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
        output = {i: forward(model, api, *inputs) for i, inputs in actual_inputs.items()}
        saved_folder = base / ENDPOINTS[name][0]
        for row in provided[name]['cases']:
            index = row['index']
            expected_pixels = read_rgb(saved_folder / f'{index}.png', row['sha256'])
            if not np.array_equal(output[index], expected_pixels):
                raise ValueError(f'Development endpoint failed to reproduce: {name}/{index}')
        endpoint_images[name] = output
    expanded, control = (endpoint_images[name] for name in ('expanded', 'C10'))
    indices = [r['index'] for r in train]

    def features(i):
        return input_features(*actual_inputs[i], expanded[i], control[i])

    x = np.stack([features(i) for i in indices])
    location, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    normalize = lambda value: torch.from_numpy(((value - location) / scale).astype(np.float32))
    fit_input = normalize(x)
    end_a, end_b, gt = (pool(values) for values in (
        [expanded[i] for i in indices], [control[i] for i in indices], [image(r['gt']) for r in train]))
    fits = {key: fit(fit_input, end_a, end_b, gt, flag) for key, flag in (('constant', False), ('adaptive', True))}
    out.mkdir(exist_ok=False)
    freeze = dump(out / 'fit.json', dict(arms=fits, training_indices=indices, mean=location.tolist(),
                                            std=scale.tolist()))
    synthetic = np.full((2, 3, 3), 128, np.uint8)
    rgb, _ = render(synthetic, synthetic.copy(), synthetic - 4, synthetic + 8, [.4, .9])
    np.testing.assert_array_equal(rgb, synthetic)
    results, predictions, artifacts = {}, {}, {}
    for label, weights in fits.items():
        folder = out / label
        folder.mkdir()
        table, pixels = {}, {}
        bias, slope = (torch.tensor(weights[name], dtype=torch.float32) for name in ('bias', 'slope'))
        for row in development:
            index = row['index']
            score = (bias + normalize(features(index)) @ slope).sigmoid().numpy()
            rgb, clipped = render(*actual_inputs[index], expanded[index], control[index], score)
            path = folder / f'{index}.png'
            Image.fromarray(rgb).save(path)
            table[index] = dict(weights=score.tolist(), sha256=digest(path), clipped_channels=clipped)
            pixels[index] = rgb
        artifacts[label], predictions[label] = table, pixels
    freeze_predictions = dump(out / 'predictions.json', artifacts)
    scores = {name: score_outputs(development, prediction, {r['index']: image(r['gt']) for r in development})
              for name, prediction in predictions.items()}
    for name, result in scores.items():
        results[name] = dict(mae=result['mean'], delta=result['delta_mean'],
                             gates={key: response_development_gate(provided['current13'], result, provided[key])
                                    for key in ('C10', 'control')})
    test = results['adaptive']
    passed = (test['mae'] < min(results['constant']['mae'], provided['expanded']['mean'])
              and all(r['proceed_to_independent_evaluation'] for r in test['gates'].values()))
    elapsed = time.monotonic() - started
    if elapsed >= 180:
        raise TimeoutError('Mixture budget exceeded')
    source = Path(__file__)
    report = dict(results=results, passed=passed, elapsed=elapsed, fit_sha256=freeze,
                  predictions_sha256=freeze_predictions, cohort_sha256=COHORT_SHA,
                  audit_sha256=AUDIT_SHA, source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'adaptive-component-mixture-protocol.md')))
    dump(out / 'results.json', report)
    print(json.dumps(dict(results={name: {'mae': r['mae'], 'delta': r['delta']} for name, r in results.items()},
                          passed=passed, seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
