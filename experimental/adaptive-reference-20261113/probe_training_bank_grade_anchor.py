"""Check a fixed reference bank as a two-image estimate of shared model offsets."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_forward
from appearance_replication import initialize_replication_model
from audit_consistent_reference import score_outputs
from independent_coco import ROOT, digest
from pairwise_gate import response_development_gate
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb
from spatial_render import checkpoint_for
from train_triplet_pilot import author_model


MODELS = {
    'expanded': ('training-coverage-expanded-v2/epoch-4.pth',
                 'cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1'),
    'C10': ('appearance-pairwise-replica-v1/epoch-7.pth',
            'abd4757eff1c5133101732c1f04a051ce29400358282baacf7a43030a5584e38'),
}


def choose_bank(train, held):
    families = {}
    for row in sorted(train, key=lambda item: item['index']):
        families.setdefault(row['reference_family'], row)
    ordering = list(families)
    if len(ordering) < 8:
        raise ValueError('Insufficient training-reference families')
    positions = np.rint(np.linspace(0, len(ordering) - 1, 8)).astype(int)
    selected = [families[ordering[i]] for i in positions]
    used = {row[key] for row in held for key in ('content_family', 'reference_family')}
    if used & {r['reference_family'] for r in selected}:
        raise ValueError('Training bank overlaps a development family')
    return selected


def apply_offset(target, reference, prediction, offset, divisor):
    if np.array_equal(target, reference):
        return target.copy(), 0
    values = np.rint(prediction + offset.astype(np.float64) / divisor)
    clipped = int(np.count_nonzero((values < 0) | (values > 255)))
    return values.clip(0, 255).astype(np.uint8), clipped


@torch.inference_mode()
def forward(model, api, target, reference):
    def tensor(rgb):
        return (torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255).cuda()
    raw, _ = appearance_forward(model, api, tensor(target), tensor(reference))
    return np.rint(raw[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return digest(path)


def produce(model, api, corpus, targets, references, rows, frozen, bank):
    totals, self_grades, predictions = {}, {}, {}
    for row in rows:
        index = row['index']
        result = forward(model, api, targets[row['group_id']], references[index])
        baseline = read_rgb(corpus / f'{index}.png', frozen[index])
        if not np.array_equal(result, baseline):
            raise ValueError(f'Baseline did not reproduce: {index}')
        predictions[index] = result
    for key, target in targets.items():
        total = np.zeros(target.shape, dtype=np.int16)
        for reference in bank:
            total += forward(model, api, target, reference)
        self_grades[key] = forward(model, api, target, target)
        totals[key] = total
    return totals, self_grades, predictions


def run(out):
    started = time.monotonic()
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    audit = checked_json(base / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(data / 'cohort.json', COHORT_SHA)
    manifest = checked_json(data / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    if not audit['passed'] or not manifest['passed']:
        raise ValueError('Incomplete frozen audit')
    entries = {row['original_path']: row for row in manifest['files']}
    rows = cohort['groups']['validation']
    if len(rows) != 64 or len({row['index'] for row in rows}) != 64:
        raise ValueError('Unexpected development group')
    fixed_scores = {name: checked_json(base / folder / 'scores.json', sha)
                    for name, (folder, sha) in ENDPOINTS.items()}

    def image(path):
        entry = entries[path]
        return read_rgb(data / entry['prepared_path'], entry['sha256'], resize=True)

    bank_rows = choose_bank(cohort['groups']['train'], rows)
    bank = [image(row['reference']) for row in bank_rows]
    bank_ids = [dict(index=row['index'], family=row['reference_family'], sha256=entries[row['reference']]['sha256'])
                for row in bank_rows]
    groups, references, targets = {}, {}, {}
    for row in rows:
        group, index = row['group_id'], row['index']
        groups.setdefault(group, []).append(index)
        if group not in targets:
            targets[group] = image(row['content'])
        references[index] = image(row['reference'])
    if len(groups) != 32 or any(len(indices) != 2 for indices in groups.values()):
        raise ValueError('Incomplete paired control')

    gray = np.full((2, 2, 3), 100, dtype=np.uint8)
    shifted, _ = apply_offset(gray, gray + 1, gray, np.zeros_like(gray) + 4, 8)
    assert np.array_equal(shifted, gray)
    shifted, _ = apply_offset(gray, gray + 1, gray + 1, np.zeros_like(gray) + 4, 8)
    assert np.array_equal(shifted, gray + 2)
    original, _ = apply_offset(gray, gray.copy(), gray + 1, np.ones_like(gray), 1)
    assert np.array_equal(original, gray)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, api, _ = author_model()
    initial_path, initial_sha = checkpoint_for('multiref-v1')
    if digest(initial_path) != initial_sha:
        raise ValueError('Frozen base changed')
    initialize_replication_model(model, torch.load(initial_path, map_location='cpu', weights_only=True))
    model.eval().requires_grad_(False)
    outputs, bank_totals, self_results = {}, {}, {}
    for name, (file, sha) in MODELS.items():
        path = base / file
        if digest(path) != sha:
            raise ValueError('Frozen trained model changed')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        pngs = {row['index']: row['sha256'] for row in fixed_scores[name]['cases']}
        bank_totals[name], self_results[name], outputs[name] = produce(
            model, api, base / ENDPOINTS[name][0], targets, references, rows, pngs, bank)

    out.mkdir(exist_ok=False)
    offsets = {
        'bank': {key: bank_totals['expanded'][key] - bank_totals['C10'][key] for key in groups},
        'self': {key: self_results['expanded'][key].astype(np.int16) - self_results['C10'][key] for key in groups},
    }
    offset_evidence, comparisons, pixels, artifacts = {}, {}, {}, {}
    for key, indices in groups.items():
        actual = sum(outputs['expanded'][i].astype(np.int16) - outputs['C10'][i] for i in indices) / 2
        comparisons[key] = dict(
            bank=float(np.abs(offsets['bank'][key] / 8 - actual).mean()),
            self=float(np.abs(offsets['self'][key] - actual).mean()),
            zero=float(np.abs(actual).mean()),
        )
        path = out / f'{key}-offsets.npz'
        np.savez(path, bank=offsets['bank'][key], self=offsets['self'][key])
        offset_evidence[key] = digest(path)
    for anchor, divisor in (('bank', 8), ('self', 1)):
        for response, sign in (('expanded', -1), ('C10', 1)):
            arm = f'{anchor}-{response}'
            folder = out / arm
            folder.mkdir()
            images, cases = {}, {}
            for row in rows:
                index, key = row['index'], row['group_id']
                rgb, clipped = apply_offset(targets[key], references[index], outputs[response][index],
                                            sign * offsets[anchor][key], divisor)
                path = folder / f'{index}.png'
                Image.fromarray(rgb).save(path)
                images[index] = rgb
                cases[index] = dict(sha256=digest(path), clipped_channels=clipped)
            pixels[arm], artifacts[arm] = images, cases
    frozen = dump(out / 'predictions.json', dict(pngs=artifacts, offsets=offset_evidence, bank=bank_ids))

    truth = {row['index']: image(row['gt']) for row in rows}
    results = {}
    for arm, images in pixels.items():
        scored = score_outputs(rows, images, truth)
        results[arm] = dict(mae=scored['mean'], response_mae=scored['delta_mean'],
                            gates={k: response_development_gate(fixed_scores['current13'], scored, fixed_scores[k])
                                   for k in ('C10', 'control')})
    offset_summary = {key: float(np.mean([item[key] for item in comparisons.values()])) for key in ('bank', 'self', 'zero')}
    elapsed = time.monotonic() - started
    if elapsed >= 180:
        raise TimeoutError('Fixed reference-bank budget exceeded')
    source = Path(__file__)
    report = dict(results=results, offset_by_target=comparisons, offset_summary=offset_summary,
                  predictions_sha256=frozen, model_sha256={k: v[1] for k, v in MODELS.items()}, base_sha256=initial_sha,
                  cohort_sha256=COHORT_SHA, audit_sha256=AUDIT_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('training-bank-grade-anchor-protocol.md')), seconds=elapsed)
    dump(out / 'results.json', report)
    print(json.dumps(dict(offset=offset_summary, arms={k: v['mae'] for k, v in results.items()}, seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
