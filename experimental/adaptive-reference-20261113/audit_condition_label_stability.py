"""CPU audit of the fixed train-only four-arm condition-label diagnosis."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import ROOT, digest


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    out = base / 'condition-label-stability-v2'
    data = base / 'cohort-multiref-v1'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    report = load(out / 'results.json')
    cohort = load(data / 'cohort.json')
    teacher_dir = base / 'condition-labels-train-v1'
    teacher = load(teacher_dir / 'results.json')
    lookup = {v['original_path']: v for v in load(data / 'srgb-v1/manifest.json')['files']}
    assert report['finished'] and report['split'] == 'train'
    assert not report['development_labels_used'] and not report['product_integrated']
    assert report['backbone_state_before'] == report['backbone_state_after']
    assert digest(teacher_dir / 'results.json') == report['teacher_results_sha256']
    assert digest(data / 'cohort.json') == report['cohort_sha256']
    assert digest(data / 'srgb-v1/manifest.json') == report['prepared_manifest_sha256']
    for name, sha in report['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == digest(out / 'frozen-source' / name) == sha
    arms = ['zero', 'zero-repeat', 'positive', 'negative']
    assert len(report['cases']) == 128
    initial = np.random.default_rng(20261128).normal(0, .5, (32, 128)).astype(np.float32)
    grouped = {}; historical = []; pngs = 0
    for arm in arms:
        cases = [c for c in report['cases'] if c['arm'] == arm]
        assert len(cases) == 32
        values = []
        for i, (case, row, old) in enumerate(zip(cases, cohort['groups']['train'][:32], teacher['cases'][:32])):
            assert (case['index'], case['group_id'], case['style_index']) == (row['index'], row['group_id'], row['style_index'])
            assert case['index'] == old['index']
            for role in ['content', 'reference', 'gt']:
                entry = lookup[row[role]]; saved = case['inputs'][role]
                assert saved['path'] == row[role]
                assert digest(data / entry['prepared_path']) == saved['parent_sha256'] == entry['sha256']
            with Image.open(data / lookup[row['gt']]['prepared_path']) as im:
                truth = np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)).astype(float)
            images = {}
            for name, key in [('baseline', 'baseline_mae'), ('gt_fit', 'fit_mae')]:
                entry = case['files'][name]; path = out / entry['path']
                assert digest(path) == entry['sha256']
                with Image.open(path) as im: images[name] = np.array(im.convert('RGB')).astype(float)
                assert images[name].shape == truth.shape == (256, 256, 3)
                assert float(np.abs(images[name] - truth).mean()) == case[key]
                pngs += 1
            entry = case['files']['conditions']; path = out / entry['path']
            assert digest(path) == entry['sha256']
            with np.load(path) as z:
                deltas = z['snapshot_deltas']; selected = z['selected_delta'].copy()
                assert deltas.shape == (9, 128) and deltas.dtype == np.float32
                assert np.isfinite(deltas).all() and np.max(np.abs(deltas)) <= 3
                assert np.array_equal(z['snapshot_steps'], np.arange(0, 129, 16))
                expected = np.zeros(128, np.float32) if arm.startswith('zero') else initial[i] * (1 if arm == 'positive' else -1)
                assert np.array_equal(deltas[0], expected)
                snapshots = case['snapshots']; best = int(np.argmin([s['mae'] for s in snapshots]))
                assert [s['step'] for s in snapshots] == list(range(0, 129, 16))
                assert case['selected_step'] == snapshots[best]['step']
                assert case['fit_mae'] == snapshots[best]['mae']
                assert case['baseline_mae'] == snapshots[0]['mae']
                assert np.array_equal(selected, deltas[best])
            if arm.startswith('zero'):
                assert case['files']['baseline']['sha256'] == old['files']['baseline']['sha256']
                assert case['selected_step'] == old['selected_step']
                with Image.open(teacher_dir / old['files']['gt_fit']['path']) as im: old_image = np.array(im).astype(float)
                with np.load(teacher_dir / old['files']['conditions']['path']) as z: old_delta = z['selected_delta'].astype(float)
                d = np.abs(old_image - images['gt_fit']); rms = float(np.sqrt(np.mean((selected.astype(float) - old_delta) ** 2)))
                assert d.mean() <= .01 and d.max() <= 1 and rms <= .001
                historical.append(dict(arm=arm, index=row['index'], output_mae=float(d.mean()), output_max=float(d.max()), parameter_rms=rms))
            values.append((case, selected.astype(float), images['gt_fit']))
        grouped[arm] = values
    assert historical == report['historical_zero_parity']
    comparisons = {}
    for arm in arms[1:]:
        records = []
        for zero, other in zip(grouped['zero'], grouped[arm]):
            distance = float(np.sqrt(np.mean((zero[1] - other[1]) ** 2)))
            image_distance = float(np.abs(zero[2] - other[2]).mean())
            score_difference = other[0]['fit_mae'] - zero[0]['fit_mae']
            records.append(dict(index=zero[0]['index'], parameter_rms=distance, output_rgb_mae=image_distance,
                                output_max_abs=float(np.abs(zero[2] - other[2]).max()), gt_mae_difference=score_difference,
                                near_output=bool(distance >= .25 and image_distance <= 1 and score_difference <= 1),
                                near_objective=bool(distance >= .25 and abs(score_difference) <= .25)))
        comparisons[arm] = records
    assert comparisons == report['comparisons']
    assert all(c['parameter_rms'] <= .001 and c['output_rgb_mae'] <= .01 and c['output_max_abs'] <= 1 for c in comparisons['zero-repeat'])
    diagnosis = {arm: dict(near_output=sum(c['near_output'] for c in cases), near_objective=sum(c['near_objective'] for c in cases)) for arm, cases in comparisons.items() if arm != 'zero-repeat'}
    assert diagnosis == report['diagnosis']
    assert report['near_output_evidence'] == any(v['near_output'] >= 8 for v in diagnosis.values())
    summary = {arm: dict(count=len(cases), initial_mean=float(np.mean([c[0]['baseline_mae'] for c in cases])), fit_mean=float(np.mean([c[0]['fit_mae'] for c in cases]))) for arm, cases in grouped.items()}
    assert summary == report['summary']
    result = dict(finished=True, producer_results_sha256=digest(out / 'results.json'), auditor_sha256=digest(Path(__file__)),
                  png_scores_checked=pngs, conditions_checked=128, snapshots_checked=1152,
                  historical_zero_checks=64, comparisons_checked=96, diagnosis=diagnosis,
                  scope='CPU saved-artifact audit; intermediate snapshots not GPU replayed')
    dst = out / 'artifact-audit.json'
    if dst.exists(): raise ValueError('Refusing overwrite')
    dst.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
