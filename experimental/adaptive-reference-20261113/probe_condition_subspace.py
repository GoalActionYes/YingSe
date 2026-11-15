"""Train-only fixed output-sensitive rank16 capacity test, not prediction."""
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from independent_coco import ROOT, digest
from train_triplet_pilot import author_model
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash
from condition_subspace import sensitivity_basis
from multiref_response import response_metrics


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261129)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-multiref-v1'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    cohort = load(data / 'cohort.json')
    manifest = load(data / 'srgb-v1/manifest.json')
    lookup = {v['original_path']: v for v in manifest['files']}
    teacher_dir = base / 'condition-labels-train-v1'
    teacher = load(teacher_dir / 'results.json')
    audit = load(teacher_dir / 'artifact-audit.json')
    assert teacher['finished'] and audit['producer_results_sha256'] == digest(teacher_dir / 'results.json')
    assert teacher['cohort_sha256'] == digest(data / 'cohort.json')
    assert teacher['prepared_manifest_sha256'] == digest(data / 'srgb-v1/manifest.json')
    history = {c['index']: c for c in teacher['cases']}
    basis_positions = list(range(0, 1024, 16))
    capacity_positions = list(range(8, 1024, 32))
    basis_rows = [cohort['groups']['train'][i] for i in basis_positions]
    rows = [cohort['groups']['train'][i] for i in capacity_positions]
    assert not {r['group_id'] for r in basis_rows} & {r['group_id'] for r in rows}
    assert all(sum(r['style_index'] == s for r in basis_rows) == 16 for s in range(4))
    assert all(sum(r['style_index'] == s for r in rows) == 8 for s in range(4))
    out = base / 'condition-subspace-v1'
    out.mkdir(exist_ok=False)
    frozen = out / 'frozen-source'; frozen.mkdir()
    names = ['probe_condition_subspace.py', 'condition-subspace-protocol.md', 'condition_subspace.py',
             'test_condition_subspace.py', 'prepared_condition.py', 'train_triplet_pilot.py',
             'check_petal_condition.py', 'training_checks.py', 'multiref_response.py']
    for name in names: (frozen / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    model, api, _ = author_model()
    checkpoint = base / 'multiref-control-v1/epoch-6.pth'
    assert digest(checkpoint) == teacher['checkpoint_sha256']
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False); before = state_hash(model)
    report = dict(scope='Fixed train-only local sensitivity/capacity, GT fitting is not inference',
                  source_sha256={n: digest(frozen / n) for n in names},
                  checkpoint_sha256=digest(checkpoint), cohort_sha256=digest(data / 'cohort.json'),
                  manifest_sha256=digest(data / 'srgb-v1/manifest.json'),
                  teacher_results_sha256=digest(teacher_dir / 'results.json'),
                  basis_positions=basis_positions, capacity_positions=capacity_positions,
                  basis_input_records=[], cases=[], product_integrated=False, prediction_model_trained=False,
                  development_used=False, backbone_state_before=before, finished=False)
    (out / 'frozen-config.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def image(row, role):
        entry = lookup[row[role]]; path = data / entry['prepared_path']
        assert digest(path) == entry['sha256']
        with Image.open(path) as im:
            array = np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
        return array, dict(path=row[role], parent_sha256=entry['sha256'])

    rng = np.random.default_rng(20261129); gradients = []
    for i, row in enumerate(basis_rows):
        arrays, records = zip(*(image(row, k) for k in ['content', 'reference']))
        c, s = [torch.from_numpy(a[None]).permute(0, 3, 1, 2).float().cuda() / 255 for a in arrays]
        cached = PreparedCondition(model, api, c, s)
        delta = torch.zeros(1, 128, device='cuda', requires_grad=True)
        small = F.interpolate(cached(delta)[0].clamp(0, 1), (16, 16), mode='area')
        probes = ((rng.integers(0, 2, size=(16, 1, 3, 16, 16), dtype=np.int8) * 2 - 1).astype(np.float32)
                  / np.sqrt(np.float32(768)))
        for j in range(16):
            gradient = torch.autograd.grad(small, delta, torch.from_numpy(probes[j]).cuda(), retain_graph=j < 15)[0]
            assert torch.isfinite(gradient).all()
            gradients.append(gradient[0].detach().cpu().numpy().copy())
        report['basis_input_records'].append(dict(index=row['index'], group_id=row['group_id'], style_index=row['style_index'], inputs=dict(zip(['content', 'reference'], records))))
        if (i + 1) % 16 == 0: print(json.dumps(dict(basis_done=i + 1)), flush=True)
    gradients = np.stack(gradients)
    values, vectors = sensitivity_basis(gradients)
    path = out / 'sensitivity-basis.npz'
    np.savez_compressed(path, gradients=gradients, eigenvalues=values, basis=vectors)
    report['basis_sha256'] = digest(path)
    report['rank16_local_energy_fraction'] = float(values[:16].sum() / values.sum())
    report['basis_saved_before_capacity_GT'] = True
    (out / 'basis-ready.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pixels = lambda raw: np.rint(raw.clip(0, 1) * 255).astype(np.uint8)
    for name, rank in [('full128', 128), ('rank16', 16)]:
        arm = out / name; arm.mkdir()
        basis = torch.from_numpy(vectors[:, :rank].astype(np.float32)).cuda()
        for offset in range(0, 32, 4):
            start = time.perf_counter(); batch = rows[offset:offset + 4]
            arrays = []; inputs = []
            for row in batch:
                values_and_records = [image(row, k) for k in ['content', 'reference', 'gt']]
                arrays.append([v[0] for v in values_and_records]); inputs.append(dict(zip(['content', 'reference', 'gt'], [v[1] for v in values_and_records])))
            c, s, truth = [torch.from_numpy(np.stack([a[k] for a in arrays])).permute(0, 3, 1, 2).float().cuda() / 255 for k in range(3)]
            cached = PreparedCondition(model, api, c, s)
            singles = [PreparedCondition(model, api, c[i:i + 1], s[i:i + 1]) for i in range(4)]
            coordinates = torch.nn.Parameter(torch.zeros(4, rank, device='cuda'))
            optimizer = torch.optim.Adam([coordinates], lr=.03)
            best = [None] * 4; traces = [[] for _ in batch]; coord_traces = [[] for _ in batch]; delta_traces = [[] for _ in batch]; baselines = [None] * 4
            for step in range(129):
                delta = coordinates @ basis.T
                if step % 16 == 0:
                    with torch.no_grad():
                        snapshots = [singles[i](delta[i:i + 1])[0][0].cpu().permute(1, 2, 0).numpy() for i in range(4)]
                        coords = coordinates.detach().cpu().numpy().copy(); ds = delta.detach().cpu().numpy().copy()
                    for i, row in enumerate(batch):
                        rgb = pixels(snapshots[i]); mae = float(np.abs(rgb.astype(float) - arrays[i][2].astype(float)).mean())
                        traces[i].append(dict(step=step, mae=mae)); coord_traces[i].append(coords[i].copy()); delta_traces[i].append(ds[i].copy())
                        if step == 0: baselines[i] = rgb.copy()
                        if best[i] is None or mae < best[i]['mae']:
                            best[i] = dict(step=step, mae=mae, image=rgb.copy(), coordinates=coords[i].copy(), delta=ds[i].copy())
                if step == 128: break
                optimizer.zero_grad(set_to_none=True)
                pred = cached(delta)[0].clamp(0, 1)
                loss = F.l1_loss(pred, truth) + .1 * (F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], truth[:, :, 1:] - truth[:, :, :-1]) + F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], truth[:, :, :, 1:] - truth[:, :, :, :-1]))
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_([coordinates], 1., error_if_nonfinite=True); optimizer.step()
                with torch.no_grad():
                    peak = (coordinates @ basis.T).abs().amax(1, keepdim=True)
                    coordinates.mul_((3 / (peak + 1e-6)).clamp(max=1))
            for i, row in enumerate(batch):
                prefix = str(row['index']); files = {}
                for key, rgb in [('baseline', baselines[i]), ('gt_fit', best[i]['image'])]:
                    path = arm / f'{prefix}-{key}.png'; Image.fromarray(rgb).save(path)
                    files[key] = dict(path=path.relative_to(out).as_posix(), sha256=digest(path))
                assert files['baseline']['sha256'] == history[row['index']]['files']['baseline']['sha256']
                path = arm / f'{prefix}-conditions.npz'
                np.savez_compressed(path, selected_coordinates=best[i]['coordinates'], selected_delta=best[i]['delta'], snapshot_coordinates=np.stack(coord_traces[i]), snapshot_deltas=np.stack(delta_traces[i]), snapshot_steps=np.arange(0, 129, 16))
                files['conditions'] = dict(path=path.relative_to(out).as_posix(), sha256=digest(path))
                report['cases'].append(dict(arm=name, rank=rank, index=row['index'], group_id=row['group_id'], style_index=row['style_index'], inputs=inputs[i], baseline_mae=traces[i][0]['mae'], fit_mae=best[i]['mae'], selected_step=best[i]['step'], snapshots=traces[i], files=files))
            (out / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(dict(arm=name, done=offset + 4, seconds=time.perf_counter() - start)), flush=True)
    groups = {}; summaries = {}
    for name in ['full128', 'rank16']:
        cases = [c for c in report['cases'] if c['arm'] == name]
        summaries[name] = dict(baseline_mean=float(np.mean([c['baseline_mae'] for c in cases])), fit_mean=float(np.mean([c['fit_mae'] for c in cases])))
        for c in cases:
            with Image.open(out / c['files']['gt_fit']['path']) as im: rgb = np.array(im.convert('RGB'))
            row = next(r for r in rows if r['index'] == c['index']); truth = image(row, 'gt')[0]
            groups.setdefault(c['group_id'], {}).setdefault(name, {})[c['style_index']] = (rgb, truth)
    paired = {name: [dict(group_id=group, styles=[a, b], **response_metrics(g[name][a][0], g[name][b][0], g[name][a][1], g[name][b][1])) for group, g in groups.items() for a, b in itertools.combinations(range(4), 2)] for name in ['full128', 'rank16']}
    full = np.array([c['fit_mae'] for c in report['cases'] if c['arm'] == 'full128']); reduced = np.array([c['fit_mae'] for c in report['cases'] if c['arm'] == 'rank16'])
    zero = summaries['full128']['baseline_mean']; retained = float((zero - reduced.mean()) / (zero - full.mean()))
    gates = dict(control_passed=bool(full.mean() <= .5 * zero), retained_gain=retained, retained_passed=bool(retained >= .9), close_count=int((reduced <= full + .5).sum()), severe=int(((reduced > full + 2) & (reduced > full * 1.25)).sum()), response_passed=bool(np.mean([r['delta_mae'] for r in paired['rank16']]) <= 1.1 * np.mean([r['delta_mae'] for r in paired['full128']])))
    gates['passed'] = bool(gates['control_passed'] and gates['retained_passed'] and gates['close_count'] >= 28 and gates['severe'] <= 2 and gates['response_passed'])
    report.update(summary=summaries, paired=paired, gates=gates, backbone_state_after=state_hash(model))
    assert before == report['backbone_state_after'] and all(p.grad is None for p in model.parameters())
    assert all(digest(Path(__file__).with_name(n)) == sha for n, sha in report['source_sha256'].items())
    report['finished'] = True
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summaries, gates=gates, energy=report['rank16_local_energy_fraction'])), flush=True)


if __name__ == '__main__': main()
