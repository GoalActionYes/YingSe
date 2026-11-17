"""Cross frozen two-reference midpoints and responses in an offline diagnostic."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from audit_consistent_reference import score_outputs
from independent_coco import ROOT, digest
from pairwise_gate import response_development_gate
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb


MODES = ('expanded', 'C10', 'current13')
ARMS = [(midpoint, response) for midpoint in MODES for response in MODES]


def compose(anchor, delta):
    middle_twice = anchor[0].astype(np.int16) + anchor[1]
    spread = delta[1].astype(np.int16) - delta[0]
    raw = [np.rint((middle_twice + sign * spread) / 2) for sign in (-1, 1)]
    outputs = [x.clip(0, 255).astype(np.uint8) for x in raw]
    actual = outputs[1].astype(np.int16) - outputs[0]
    return outputs, dict(clipped_channels=int(sum(np.count_nonzero((x < 0) | (x > 255)) for x in raw)),
                         response_perturbation=float(np.abs(actual - spread).mean()))


def split_energy(a, truth):
    e0, e1 = (x.astype(np.float64) - y for x, y in zip(a, truth, strict=True))
    common = ((e0 + e1) / 2) ** 2
    response = ((e1 - e0) / 2) ** 2
    np.testing.assert_allclose(common + response, (e0**2 + e1**2) / 2, rtol=0, atol=1e-10)
    return dict(midpoint_mse=float(common.mean()), half_response_mse=float(response.mean()))


def save(folder, index, rgb):
    path = folder / f'{index}.png'
    Image.fromarray(rgb).save(path)
    return digest(path)


def run(out):
    started = time.monotonic()
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    audit = checked_json(base / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(data / 'cohort.json', COHORT_SHA)
    prepared = checked_json(data / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    if not audit['passed'] or not prepared['passed']:
        raise ValueError('Frozen output and pairing audit not accepted')
    rows = cohort['groups']['validation']
    lookup = {row['original_path']: row for row in prepared['files']}
    groups = {}
    for row in rows:
        group = groups.setdefault(row['group_id'], {})
        if row['style_index'] in group:
            raise ValueError('Duplicate target/role')
        group[row['style_index']] = row['index']
    if len(rows) != 64 or len(groups) != 32 or any(set(v) != {0, 1} for v in groups.values()):
        raise ValueError('Unexpected development size')
    original = {name: checked_json(base / folder / 'scores.json', sha) for name, (folder, sha) in ENDPOINTS.items()}
    inputs = {}
    for name in MODES:
        folder = base / ENDPOINTS[name][0]
        inputs[name] = {r['index']: read_rgb(folder / f'{r["index"]}.png', r['sha256'])
                        for r in original[name]['cases']}

    out.mkdir(exist_ok=False)
    schedules, artifacts, corrections = {}, {}, {}
    for mid, response in ARMS:
        label = f'{mid}--{response}'
        result, diagnostics, hashes = {}, {}, {}
        folder = out / label
        folder.mkdir()
        for key, group in groups.items():
            i, j = group[0], group[1]
            images, detail = compose([inputs[mid][i], inputs[mid][j]], [inputs[response][i], inputs[response][j]])
            diagnostics[key] = detail
            for index, rgb in zip((i, j), images, strict=True):
                if mid == response and not np.array_equal(rgb, inputs[mid][index]):
                    raise ValueError('Diagonal output failed to reproduce')
                result[index] = rgb
                hashes[index] = save(folder, index, rgb)
        schedules[label], artifacts[label], corrections[label] = result, hashes, diagnostics
    freeze = out / 'renders.json'
    freeze.write_text(json.dumps(artifacts, indent=2) + '\n', encoding='utf-8')

    truth = {}
    for row in rows:
        item = lookup[row['gt']]
        truth[row['index']] = read_rgb(data / item['prepared_path'], item['sha256'], resize=True)
    scoring = {arm: score_outputs(rows, prediction, truth) for arm, prediction in schedules.items()}
    for name in MODES:
        linked = original[name]
        clean = dict(linked, cases=[{k: v for k, v in x.items() if k != 'sha256'} for x in linked['cases']])
        if scoring[f'{name}--{name}'] != clean:
            raise ValueError('Diagonal scores did not reproduce original results')
    summary = {arm: dict(mae=value['mean'], response_mae=value['delta_mean'],
                         clipped_channels=sum(row['clipped_channels'] for row in corrections[arm].values()),
                         mean_response_perturbation=float(np.mean([row['response_perturbation']
                                                                  for row in corrections[arm].values()])),
                         gates={name: response_development_gate(original['current13'], value, original[name])
                                for name in ('C10', 'control')})
               for arm, value in scoring.items()}
    row_mae = {arm: {r['index']: r['mae'] for r in value['cases']} for arm, value in scoring.items()}
    pair_mae = {arm: {r['group_id']: r['delta_mae'] for r in value['paired']} for arm, value in scoring.items()}
    external = {mode: {r['group_id']: r['delta_mae'] for r in original[mode]['paired']}
                for mode in ('C10', 'control')}
    chosen, pairing = {}, {}
    oracle_folder = out / 'oracle-min-rgb'
    oracle_folder.mkdir()
    upper_count = dict(C10=0, control=0)
    for key, group in groups.items():
        i, j = group[0], group[1]
        by_rgb = {arm: (mae[i] + mae[j]) / 2 for arm, mae in row_mae.items()}
        by_response = {arm: mae[key] for arm, mae in pair_mae.items()}
        best_rgb = min(by_rgb, key=by_rgb.get)
        best_delta = min(by_response, key=by_response.get)
        for mode in external:
            upper_count[mode] += int(external[mode][key] - by_response[best_delta] > .1)
        output = {}
        for index in (i, j):
            chosen[index] = schedules[best_rgb][index]
            output[index] = save(oracle_folder, index, chosen[index])
        energies = {name: split_energy([inputs[name][i], inputs[name][j]], [truth[i], truth[j]]) for name in MODES}
        pairing[key] = dict(rgb_mae=by_rgb, delta_mae=by_response, rgb_choice=best_rgb, best_delta_choice=best_delta,
                            energy=energies, oracle_sha256=output)
    result = score_outputs(rows, chosen, truth)
    oracle = dict(mean=result['mean'], delta_mean=result['delta_mean'],
                  gates={name: response_development_gate(original['current13'], result, original[name])
                         for name in external})
    elapsed = time.monotonic() - started
    if elapsed >= 90:
        raise TimeoutError('Frozen CPU budget exceeded')
    script = Path(__file__)
    report = dict(modes=summary, pairings=pairing, rgb_oracle=oracle, best_possible_improved_delta_count=upper_count,
                  elapsed_seconds=elapsed, renders_sha256=digest(freeze), cohort_sha256=COHORT_SHA,
                  audit_sha256=AUDIT_SHA, endpoints={name: sha for name, (_, sha) in ENDPOINTS.items()},
                  source_sha256=digest(script), protocol_sha256=digest(script.with_name(
                      'paired-grade-components-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(oracle_mae=oracle['mean'], count=upper_count, seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
