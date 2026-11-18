"""Bound an offline scalar cold-axis change of the frozen local grade."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import LUMA, edit_weight, target_regions
from independent_coco import ROOT, digest
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import COOL, ROLES, compare, gates_for, mask_for
from probe_reference_chroma_spread import BASE, PARENT


REDISTRIBUTED_SHA = 'a80c840cceb6067dd79a63e98ddf6c1180968ff27a0635deede672960c2c0c82'
THRESHOLDS = {'person': 1.5, 'neutral_person': 1.5, 'edited_core': 3.0}


def optimal_axis(source, desired):
    source = source.astype(float)
    ratios = (desired - source) / COOL
    order = np.argsort(ratios, axis=-1)
    cumulative = np.cumsum(np.abs(COOL)[order], axis=-1)
    middle = (cumulative < np.abs(COOL).sum()/2).sum(axis=-1)
    sorted_ratios = np.take_along_axis(ratios, order, axis=-1)
    scalar = np.take_along_axis(sorted_ratios, middle[..., None], axis=-1).squeeze(-1)
    limits = np.stack((-source / COOL, (255-source) / COOL))
    low = np.min(limits, axis=0).max(axis=-1)
    high = np.max(limits, axis=0).min(axis=-1)
    raw = source + np.clip(scalar, low, high)[..., None]*COOL
    np.testing.assert_allclose((raw-source) @ LUMA, 0, atol=1e-12)
    assert raw.min() >= -1e-12 and raw.max() <= 255+1e-12
    return raw, np.rint(raw.clip(0, 255)).astype(np.uint8)


def run(out):
    docs = ROOT/'docs/research'
    toggle = docs/'2026-11-18-local-edit-toggle-evidence.json'
    redistributed = docs/'2026-11-18-local-chroma-redistribution-evidence.json'
    assert digest(toggle) == PARENT_REPORT and digest(redistributed) == REDISTRIBUTED_SHA
    old, comparison = [json.loads(path.read_bytes()) for path in (toggle, redistributed)]
    parent_path = PARENT/'results.json'
    assert digest(parent_path) == old['parent_sha256']
    parent = json.loads(parent_path.read_bytes())
    entries = {case['pair']: case for case in parent['cases']}
    probe = np.array([[[112, 107, 98]]], dtype=np.uint8)
    desired = probe + COOL
    np.testing.assert_allclose(optimal_axis(probe, desired)[0], desired, atol=1e-12)
    assert comparison['toggle_sha256'] == PARENT_REPORT
    out.mkdir()
    cases = {}
    for pair, evidence in old['cases'].items():
        folder = BASE/'local-edit-toggle-v1'/pair

        def checked(name):
            path = folder/name
            assert digest(path) == evidence['files'][name]
            with Image.open(path) as image:
                return np.array(image)

        target = checked('target.png')
        current = {mode: checked(f'{mode}-current13.png') for mode in ROLES}
        expected = {mode: checked(f'{mode}-expected.png') for mode in ROLES}
        humans = tuple(mask_for(entries[pair], photo) for photo in pair.split('-'))
        weights = tuple(edit_weight(mask, 1)[0] if mask.any() else np.zeros(mask.shape) for mask in humans)
        masks = target_regions(target, humans[0])
        masks.update(edited_core=weights[0] == 1, untouched=weights[0] == 0,
                     neutral_person=masks['neutral_proxy'] & humans[0],
                     neutral_edited=masks['neutral_proxy'] & (weights[0] == 1))
        desired_local = current['base'].astype(float) + expected['local'].astype(float) - expected['base']
        raw, repaired = optimal_axis(current['local'], desired_local)
        if not humans[0].any():
            assert np.array_equal(current['base'], current['local'])
            assert np.array_equal(repaired, current['local'])
        image_path = out/f'{pair}.png'
        Image.fromarray(repaired).save(image_path)
        actual = dict(current, local=repaired)
        detail = compare(actual, expected, masks)
        gates = gates_for(detail, actual, humans, weights)
        residual = np.abs(raw-desired_local)
        bounds = {}
        for name, threshold in THRESHOLDS.items():
            area = masks[name]
            if not area.any():
                continue
            continuous = float(residual[area].mean())
            integer_floor = max(0., continuous - .5)
            qualified = gates['positive'] and (name != 'neutral_person' or int(area.sum()) >= 64)
            bounds[name] = dict(continuous=continuous, quantized_lower_bound=integer_floor,
                                threshold=threshold, excludes_passing=bool(qualified and integer_floor > threshold),
                                achieved=detail['local'][name]['error'],
                                baseline=evidence['metrics']['local'][name]['error'],
                                redistributed=comparison['cases'][pair]['metrics']['local'][name]['error'])
        local = {name: val for name, val in gates['checks'].items() if name.startswith('local') and val is not None}
        cases[pair] = dict(bounds=bounds, projection=(detail['local']['edited_core'] or {}).get('projection'),
                           gates=gates, local_passed=bool(local) and all(local), sha256=digest(image_path))

    excluded = [key for key, row in cases.items() if any(v['excludes_passing'] for v in row['bounds'].values())]
    passing = [key for key, row in cases.items() if row['local_passed']]
    result = dict(cases=cases, excluded=excluded, passing=passing, toggle_sha256=PARENT_REPORT,
                  redistributed_sha256=REDISTRIBUTED_SHA, source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(excluded=excluded, passing=passing)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
