"""Screen a fixed inverse color-transfer signal on consumed paired outputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from audit_consistent_reference import score_outputs
from independent_coco import ROOT, digest
from pairwise_gate import response_development_gate
from spatial_render import SpatialEngine


COHORT_SHA = 'ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
AUDIT_SHA = 'e9597b1640962acb0837306d4eae38c0e069cd86e322c95f7d1e43ce62d6b1a0'
ENDPOINTS = {
    'current13': ('appearance-pairwise-replica-v1/baseline-current',
                  '1da758af9e93e565b114eec48644982e7db23c6bf2760785c34242fe3c349094'),
    'expanded': ('training-coverage-expanded-v2/epoch-4',
                 '2f52cf0179fef1179261449afcbd29207abe25d1d979a41b5606381c3be13558'),
    'control': ('training-coverage-control-v2/epoch-1',
                '42d983eaa9fb9b2390bc1cfbfd250233657ee41ea29344b79e4b23b741bafb0e'),
    'C10': ('appearance-pairwise-replica-v1/epoch-7',
            '05ef15fe3327b09558e70067fa6dc22bc85190426513aecc5841f72bd7ede1b6'),
}
FORWARD = ('current13', 'expanded')


def choose(scores):
    first, second = (scores[name] for name in FORWARD)
    if first['ratio'] is not None and second['ratio'] is not None and second['ratio'] < first['ratio']:
        return 'expanded'
    return 'current13'


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Frozen artifact changed: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def read_rgb(path, sha, *, resize=False):
    if digest(path) != sha:
        raise ValueError(f'Frozen RGB changed: {path.name}')
    with Image.open(path) as image:
        if resize:
            return np.asarray(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))
        if image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError('Unexpected frozen endpoint format')
        return np.array(image)


def count_selection(selected, fixed, rows):
    base = np.array([x['mae'] for x in fixed['current13']['cases']])
    original = np.array([x['mae'] for x in fixed['expanded']['cases']])
    chosen = np.array([x['mae'] for x in selected['cases']])
    severe = (original > base + 5) & (original > 1.25 * base)
    kept = (chosen > base + 5) & (chosen > 1.25 * base)
    return dict(expanded_chosen=sum(r['chosen'] == 'expanded' for r in rows),
                expanded_beneficial_rejected=int(((original < base - .1) & (chosen == base)).sum()),
                improves_over_current=int((chosen < base - .1).sum()), worsens_current=int((chosen > base + .1).sum()),
                severe_before=[r['index'] for r, flag in zip(rows, severe) if flag],
                severe_after=[r['index'] for r, flag in zip(rows, kept) if flag])


def run(out):
    start = time.monotonic()
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    audit = checked_json(base / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(data / 'cohort.json', COHORT_SHA)
    prepared = checked_json(data / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    if not audit['passed'] or not prepared['passed']:
        raise ValueError('Missing successful input audit')
    lookup = {row['original_path']: row for row in prepared['files']}
    rows = cohort['groups']['validation']
    if len(rows) != 64 or len({row['index'] for row in rows}) != 64:
        raise ValueError('Unexpected development cohort')
    scores = {name: checked_json(base / folder / 'scores.json', sha) for name, (folder, sha) in ENDPOINTS.items()}
    checksums = {mode: {r['index']: r['sha256'] for r in scores[mode]['cases']} for mode in FORWARD}
    assert choose(dict(current13={'ratio': .5}, expanded={'ratio': .5})) == 'current13'
    assert choose(dict(current13={'ratio': None}, expanded={'ratio': .1})) == 'current13'
    assert choose(dict(current13={'ratio': .7}, expanded={'ratio': .5})) == 'expanded'
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    out.mkdir(exist_ok=False)
    chosen, weak, records = {}, {}, []
    for row in rows:
        case = row['index']
        source = lookup[row['content']]
        original = read_rgb(data / source['prepared_path'], source['sha256'], resize=True)
        observations, forward, paths = {}, {}, {}
        for name in FORWARD:
            folder, _ = ENDPOINTS[name]
            paths[name] = base / folder / f'{case}.png'
            forward[name] = grade = read_rgb(paths[name], checksums[name][case])
            reverse = engine.prepare(grade, original).apply(grade, 1.)
            magnitude = mae(original, grade)
            error = mae(original, reverse)
            reverse_path = out / f'{case}-{name}-reverse.png'
            Image.fromarray(reverse).save(reverse_path)
            observations[name] = dict(magnitude=magnitude, inverse=error,
                                      ratio=error / magnitude if magnitude >= 1 else None,
                                      reverse_sha256=digest(reverse_path), endpoint_sha256=checksums[name][case])
        winner = choose(observations)
        less_change = min(FORWARD, key=lambda name: observations[name]['magnitude'])
        chosen[case], weak[case] = forward[winner], forward[less_change]
        selected = out / f'{case}-selected.png'
        selected.write_bytes(paths[winner].read_bytes())
        if digest(selected) != checksums[winner][case]:
            raise ValueError('Selected endpoint identity changed')
        records.append(dict(index=case, chosen=winner, least_change=less_change, endpoints=observations,
                            selected_sha256=checksums[winner][case], content_sha256=source['sha256']))
    frozen = out / 'choices.json'
    frozen.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    truths = {}
    for row in rows:
        entry = lookup[row['gt']]
        truths[row['index']] = read_rgb(data / entry['prepared_path'], entry['sha256'], resize=True)
    candidate = score_outputs(rows, chosen, truths)
    control = score_outputs(rows, weak, truths)
    selection = count_selection(candidate, scores, records)
    measurement = dict(original_C10=response_development_gate(scores['current13'], candidate, scores['C10']),
                       old_continuation=response_development_gate(scores['current13'], candidate, scores['control']))
    keep_quality = candidate['mean'] <= scores['expanded']['mean'] and (
        len(selection['severe_after']) < len(selection['severe_before']))
    passed = keep_quality and all(row['proceed_to_independent_evaluation'] for row in measurement.values())
    elapsed = time.monotonic() - start
    if elapsed >= 120:
        raise TimeoutError('Frozen inverse-screen budget exceeded')
    source = Path(__file__)
    report = dict(cases=records, outcomes=selection, gates=measurement, passed=passed, seconds=elapsed,
                  selected=candidate, least_change=control, choices_sha256=digest(frozen),
                  endpoint_scores={k: v[1] for k, v in ENDPOINTS.items()}, inverse_model_sha256=engine.checkpoint_sha,
                  cohort_sha256=COHORT_SHA, audit_sha256=AUDIT_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('roundtrip-grade-choice-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(selected=candidate['mean'], weakest=control['mean'], elapsed=elapsed,
                          outcomes=selection, passed=passed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
