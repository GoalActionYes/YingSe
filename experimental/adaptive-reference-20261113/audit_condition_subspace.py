"""CPU audit of saved sensitivity basis and known-GT capacity artifacts."""
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import ROOT, digest
from multiref_response import response_metrics


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    out = base / 'condition-subspace-v1'; data = base / 'cohort-multiref-v1'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    r = load(out / 'results.json'); ready = load(out / 'basis-ready.json')
    cohort = load(data / 'cohort.json'); manifest = load(data / 'srgb-v1/manifest.json')
    lookup = {v['original_path']: v for v in manifest['files']}
    teacher_dir = base / 'condition-labels-train-v1'; teacher = load(teacher_dir / 'results.json')
    history = {c['index']: c for c in teacher['cases']}
    assert r['finished'] and not r['product_integrated'] and not r['prediction_model_trained'] and not r['development_used']
    assert r['backbone_state_before'] == r['backbone_state_after']
    assert digest(data / 'cohort.json') == r['cohort_sha256']
    assert digest(data / 'srgb-v1/manifest.json') == r['manifest_sha256']
    assert digest(teacher_dir / 'results.json') == r['teacher_results_sha256']
    for name, sha in r['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == digest(out / 'frozen-source' / name) == sha
    assert r['basis_positions'] == list(range(0, 1024, 16))
    assert r['capacity_positions'] == list(range(8, 1024, 32))
    basis_rows = [cohort['groups']['train'][i] for i in r['basis_positions']]
    rows = [cohort['groups']['train'][i] for i in r['capacity_positions']]
    assert not {c['group_id'] for c in basis_rows} & {c['group_id'] for c in rows}
    assert ready['cases'] == [] and ready['basis_saved_before_capacity_GT']
    assert ready['basis_sha256'] == r['basis_sha256'] == digest(out / 'sensitivity-basis.npz')
    assert len(r['basis_input_records']) == 64
    for row, saved in zip(basis_rows, r['basis_input_records']):
        assert saved['index'] == row['index'] and set(saved['inputs']) == {'content', 'reference'}
        for role, entry in saved['inputs'].items():
            assert entry['path'] == row[role]
            p = lookup[row[role]]
            assert digest(data / p['prepared_path']) == p['sha256'] == entry['parent_sha256']
    with np.load(out / 'sensitivity-basis.npz') as z:
        g = z['gradients'].astype(float); values = z['eigenvalues']; basis = z['basis']
    assert g.shape == (1024, 128) and np.isfinite(g).all()
    assert values.shape == (128,) and basis.shape == (128, 128)
    covariance = g.T @ g / 1024
    np.testing.assert_allclose(basis.T @ basis, np.eye(128), atol=1e-12)
    np.testing.assert_allclose(covariance @ basis, basis * values[None], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(covariance)[::-1], rtol=1e-7, atol=1e-12)
    assert np.all(np.diff(values) <= 0)
    assert all(basis[np.argmax(np.abs(basis[:, j])), j] >= 0 for j in range(128))
    assert float(values[:16].sum() / values.sum()) == r['rank16_local_energy_fraction']
    assert len(r['cases']) == 64
    grouped = {}; summaries = {}; pngs = 0
    for arm, rank in [('full128', 128), ('rank16', 16)]:
        cases = [c for c in r['cases'] if c['arm'] == arm]
        assert len(cases) == 32
        for row, c in zip(rows, cases):
            assert (row['index'], row['group_id'], row['style_index']) == (c['index'], c['group_id'], c['style_index'])
            assert c['rank'] == rank
            for role, saved in c['inputs'].items():
                p = lookup[row[role]]
                assert saved['path'] == row[role] and saved['parent_sha256'] == p['sha256'] == digest(data / p['prepared_path'])
            with Image.open(data / lookup[row['gt']]['prepared_path']) as im:
                truth = np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
            images = {}
            for name, key in [('baseline', 'baseline_mae'), ('gt_fit', 'fit_mae')]:
                entry = c['files'][name]; p = out / entry['path']; assert digest(p) == entry['sha256']
                with Image.open(p) as im: images[name] = np.array(im.convert('RGB'))
                assert images[name].shape == (256, 256, 3)
                assert float(np.abs(images[name].astype(float) - truth.astype(float)).mean()) == c[key]
                pngs += 1
            assert c['files']['baseline']['sha256'] == history[row['index']]['files']['baseline']['sha256']
            entry = c['files']['conditions']; p = out / entry['path']; assert digest(p) == entry['sha256']
            with np.load(p) as z:
                coords = z['snapshot_coordinates']; deltas = z['snapshot_deltas']
                assert coords.shape == (9, rank) and deltas.shape == (9, 128)
                assert np.isfinite(coords).all() and np.isfinite(deltas).all() and np.abs(deltas).max() <= 3
                np.testing.assert_allclose(coords.astype(float) @ basis[:, :rank].T, deltas, rtol=1e-5, atol=2e-6)
                assert not coords[0].any() and not deltas[0].any()
                assert np.array_equal(z['snapshot_steps'], np.arange(0, 129, 16))
                best = int(np.argmin([v['mae'] for v in c['snapshots']]))
                assert [v['step'] for v in c['snapshots']] == list(range(0, 129, 16))
                assert c['selected_step'] == c['snapshots'][best]['step']
                assert c['fit_mae'] == c['snapshots'][best]['mae'] and c['baseline_mae'] == c['snapshots'][0]['mae']
                assert np.array_equal(z['selected_delta'], deltas[best])
                assert np.array_equal(z['selected_coordinates'], coords[best])
            grouped.setdefault(row['group_id'], {}).setdefault(arm, {})[row['style_index']] = (images['gt_fit'], truth)
        summaries[arm] = dict(baseline_mean=float(np.mean([c['baseline_mae'] for c in cases])), fit_mean=float(np.mean([c['fit_mae'] for c in cases])))
    paired = {name: [dict(group_id=group, styles=[a, b], **response_metrics(g[name][a][0], g[name][b][0], g[name][a][1], g[name][b][1])) for group, g in grouped.items() for a, b in itertools.combinations(range(4), 2)] for name in ['full128', 'rank16']}
    assert paired == r['paired'] and summaries == r['summary']
    full = np.array([c['fit_mae'] for c in r['cases'] if c['arm'] == 'full128'])
    reduced = np.array([c['fit_mae'] for c in r['cases'] if c['arm'] == 'rank16'])
    zero = summaries['full128']['baseline_mean']; retained = float((zero - reduced.mean()) / (zero - full.mean()))
    gates = dict(control_passed=bool(full.mean() <= .5 * zero), retained_gain=retained, retained_passed=bool(retained >= .9), close_count=int((reduced <= full + .5).sum()), severe=int(((reduced > full + 2) & (reduced > full * 1.25)).sum()), response_passed=bool(np.mean([c['delta_mae'] for c in paired['rank16']]) <= 1.1 * np.mean([c['delta_mae'] for c in paired['full128']])))
    gates['passed'] = bool(gates['control_passed'] and gates['retained_passed'] and gates['close_count'] >= 28 and gates['severe'] <= 2 and gates['response_passed'])
    assert gates == r['gates']
    result = dict(finished=True, producer_results_sha256=digest(out / 'results.json'), auditor_sha256=digest(Path(__file__)), basis_records_checked=64, png_scores_checked=pngs, condition_files_checked=64, snapshots_checked=576, response_records_checked=96, gates=gates, scope='CPU saved artifacts; no VJP extraction, optimization or GPU snapshot replay')
    dst = out / 'artifact-audit.json'
    if dst.exists(): raise ValueError('Refusing overwrite')
    dst.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
