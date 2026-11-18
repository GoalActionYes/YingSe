"""Learn a chroma correction with and without reference detail statistics."""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from audit_tst_reference_equivalence import COHORT, HASHES, MANIFEST
from background_intervention import LUMA, target_regions
from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import save
from probe_neutral_relation_balance import controlled as local_controls
from probe_reference_chroma_spread import PARENT, ROOT, decision as background_decision, projection
from probe_roundtrip_grade_choice import checked_json, read_rgb
from spatial_render import SpatialEngine, finish_spatial
from train_tst_reference_equivalence import FROZEN_BASELINE, SPLIT, SPLIT_SHA, decision


BASELINE_SHA = '44b8c73ec404d8b69d62a5399e3057d6c528c41b5241365469e673fcf1e95351'
LABELS = ('same_truth', 'same_truth', 'different_truth')
BANDS = (np.arange(8) + .5) * 32


def describe(rgb):
    rgb = rgb.astype(np.float64)
    y = rgb @ LUMA
    c = rgb - y[..., None]
    color = np.concatenate(([y.mean(), y.std()], c.mean((0, 1)), c.std((0, 1))))
    details = []
    for step in (1, 8):
        dy = np.concatenate(((y[step:] - y[:-step]).ravel(), (y[:, step:] - y[:, :-step]).ravel()))
        dc = np.concatenate(((c[step:] - c[:-step]).reshape(-1, 3),
                             (c[:, step:] - c[:, :-step]).reshape(-1, 3)))
        details.extend((np.abs(dy).mean(), *np.abs(dc).mean(0),
                        *((dy @ dc) / max(float(dy @ dy), 1e-12))))
    return color, np.asarray(details)


def stats(target, reference):
    tc, td = describe(target)
    rc, rd = describe(reference)
    return np.concatenate((tc, rc)), np.concatenate((tc, rc, td, rd))


def labels(target, truth, baseline):
    delta = truth.astype(np.float64) - baseline
    delta -= (delta @ LUMA)[..., None]
    group = np.floor((target.astype(float) @ LUMA) / 32).astype(int).clip(0, 7)
    counts = np.bincount(group.ravel(), minlength=8)
    values = np.stack([np.bincount(group.ravel(), weights=delta[..., i].ravel(), minlength=8)
                       for i in range(3)], axis=1)
    return values / counts.clip(1)[:, None], counts / group.size


def fit(features, residuals, frequency):
    mean = features.mean(0)
    scale = features.std(0).clip(1e-6)
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.eye(design.shape[1]) * .1
    penalty[0, 0] = 0
    coefficients = []
    for band in range(8):
        weights = frequency[:, band]
        assert weights.sum() > 0
        weights = weights / weights.sum()
        weighted = design.T * weights
        system = weighted @ design + penalty
        rhs = weighted @ residuals[:, band]
        solution = np.linalg.solve(system, rhs)
        np.testing.assert_allclose(system @ solution, rhs, atol=1e-9, rtol=0)
        coefficients.append(solution)
    return mean, scale, np.stack(coefficients, axis=1)


def render(target, baseline, feature, model):
    mean, scale, coefficients = model
    x = np.r_[1, (feature - mean) / scale]
    vectors = np.einsum('i,ijk->jk', x, coefficients)
    vectors -= (vectors @ LUMA)[:, None]
    vectors *= np.divide(8., np.max(np.abs(vectors), axis=1).clip(8.))[:, None]
    luminosity = target.astype(float) @ LUMA
    correction = np.stack([np.interp(luminosity, BANDS, vectors[:, i]) for i in range(3)], axis=2)
    np.testing.assert_allclose(correction @ LUMA, 0, atol=1e-12, rtol=0)
    raw = baseline + correction
    return np.rint(raw.clip(0, 255)).astype(np.uint8), dict(
        applied=True, max_offset=float(np.abs(correction).max()),
        clipped=float(((raw < 0) | (raw > 255)).mean())
    )


class Candidate:
    def __init__(self, model):
        self.model = model

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(applied=False)
        return render(target, baseline, stats(target, reference)[1], self.model)


def score(target, truths, rgb, index):
    masks = target_regions(target, np.zeros(target.shape[:2], bool))
    first, other, different = (image.astype(np.int16) for image in rgb)
    expected = truths[1] - truths[0]
    out = {}
    for key in ('whole', 'neutral_proxy'):
        mask = masks[key]
        errors = [float(np.abs((image - truth)[mask]).mean())
                  for image, truth in zip((first, other, different), (truths[0], truths[0], truths[1]), strict=True)]
        changes = different - first, different - other
        out[key] = dict(expected_delta=float(np.abs(expected[mask]).mean()),
                        same_swap=float(np.abs((first - other)[mask]).mean()), errors=errors,
                        different_swap=[float(np.abs(change[mask]).mean()) for change in changes],
                        projection=projection((changes[0] + changes[1]) / 2, expected, mask))
    whole = out['whole']
    passed = whole['expected_delta'] >= 1 and whole['same_swap'] < min(whole['different_swap'])
    return dict(index=index, passed=bool(passed and whole['projection'] >= .50), regions=out)


def run(out):
    start = time.monotonic()
    torch.set_num_threads(4)
    rows = checked_json(SPLIT / 'private-assignment.json', SPLIT_SHA)
    assert len(rows['training']) == 82 and len(rows['development']) == 24
    original = checked_json(FROZEN_BASELINE, BASELINE_SHA)
    assert original['metadata']['assignment_sha256'] == SPLIT_SHA
    asset_index = {row['original_path']: row for row in checked_json(MANIFEST, HASHES[2])['files']}

    @lru_cache(maxsize=512)
    def image(key):
        asset = asset_index[key]
        path = COHORT.parent / asset['prepared_path']
        assert digest(path) == asset['sha256']
        with Image.open(path) as im:
            return np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))

    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == original['checkpoint_sha256']
    out.mkdir(exist_ok=False)
    features, errors, frequencies = [], [], []
    for case in rows['training']:
        target = image(case['content'])
        for key, label in zip(case['refs'], LABELS, strict=True):
            reference = image(key)
            baseline = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
            truth = finish_spatial(target, image(case[label]), .8)
            error, frequency = labels(target, truth, baseline)
            errors.append(error)
            frequencies.append(frequency)
            features.append(stats(target, reference))
    models = [fit(np.stack(values), np.array(errors), np.array(frequencies)) for values in zip(*features, strict=True)]
    names = ('color', 'detail')
    hashes = {name: {} for name in names}
    for index, row in enumerate(rows['development']):
        target = image(row['content'])
        for slot, key in enumerate(row['refs']):
            file = f'{index}-{slot}.png'
            baseline = read_rgb(FROZEN_BASELINE.parent / file, original['predictions'][file])
            for name, model, feature in zip(names, models, stats(target, image(key)), strict=True):
                output, _ = render(target, baseline, feature, model)
                hashes[name][file] = save(out / f'{name}-{file}', output)
    predictions = out / 'predictions.json'
    predictions.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    reports = {name: dict(cases=[]) for name in names}
    for index, row in enumerate(rows['development']):
        target = image(row['content'])
        gt = [finish_spatial(target, image(row[key]), .8).astype(np.int16)
              for key in ('same_truth', 'different_truth')]
        for name in names:
            results = [read_rgb(out / f'{name}-{index}-{slot}.png', hashes[name][f'{index}-{slot}.png'])
                       for slot in range(3)]
            reports[name]['cases'].append(score(target, gt, results, index))
    for report in reports.values():
        report['passed_cases'] = sum(row['passed'] for row in report['cases'])

    manifest = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(ROOT / 'docs/research/2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    assert toggle['parent_sha256'] == PARENT_SHA
    probe = Candidate(models[1])
    flat = np.full((32, 32, 3), 127, np.uint8)
    unchanged, _ = probe.measure(flat, flat.copy(), flat - 8)
    np.testing.assert_array_equal(flat, unchanged)
    flat_color, flat_detail = stats(flat, flat + 3)
    assert np.isfinite(flat_color).all() and not flat_detail[flat_color.size:].any()
    background = background_controls(out, manifest, probe.measure)
    local = local_controls(probe, out, toggle, {r['pair']: r for r in manifest['cases']})
    elapsed = time.monotonic() - start
    assert elapsed <= 150
    source = Path(__file__)
    result = dict(arms=reports, comparisons=decision(original, reports['color'], reports['detail']),
                  background=background, background_gate=background_decision(background), local=local,
                  same_image=True, seconds=elapsed, fit_rows=len(features),
                  predictions_sha256=digest(predictions), baseline_sha256=BASELINE_SHA,
                  assignment_sha256=SPLIT_SHA, source_sha256=digest(source),
                  weights_sha256=engine.checkpoint_sha, author_manifest_sha256=HASHES[2],
                  background_sha256=PARENT_SHA, toggle_sha256=PARENT_REPORT,
                  protocol_sha256=digest(source.with_name('reference-detail-color-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(comparisons=result['comparisons'], background=result['background_gate'],
                          seconds=elapsed)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
