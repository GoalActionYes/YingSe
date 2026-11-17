"""Offline replay of a conserved cold-color edit with a known target region."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from background_intervention import LUMA, edit_weight, target_regions
from independent_coco import digest
from probe_local_edit_toggle import COOL, ROLES, compare, gates_for, mask_for, save
from probe_reference_chroma_spread import BASE, PARENT


PARENT_REPORT = '437681478dfde0362c953b0210598fbc4f23f1d043e7be8514f4a49d4df5857a'


def redistribute(base, local, weight):
    total_weight = weight.sum()
    if total_weight <= 0:
        return local.copy(), dict(applied=False)
    offset = COOL / np.linalg.norm(COOL)
    original = local.astype(np.float64)
    coefficients = (original - base) @ offset
    removed = coefficients * (1 - weight)
    shift = weight * (removed.sum() / total_weight) - removed
    raw = original + shift[..., None] * offset
    np.testing.assert_allclose((raw - original) @ LUMA, 0, atol=1e-12)
    np.testing.assert_allclose((raw - original).sum(axis=(0, 1)), 0, atol=1e-7)
    rendered = np.rint(raw.clip(0, 255)).astype(np.uint8)
    final = rendered.astype(np.float64) - original
    return rendered, dict(applied=True, removed_axis_total=float(removed.sum()),
                          axis_total_before=float(coefficients.sum()),
                          axis_total_after_quantization=float(((rendered.astype(np.float64) - base) @ offset).sum()),
                          clipped_channel_fraction=float(((raw < 0) | (raw > 255)).mean()),
                          quantized_luma_mae=float(np.abs(final @ LUMA).mean()),
                          max_quantized_luma_change=float(np.abs(final @ LUMA).max()))


def synthetic_check():
    original = np.full((1, 2, 3), 128, dtype=np.uint8)
    tagged = original.copy()
    tagged[0, 0] = [120, 129, 145]
    weighted, diagnostics = redistribute(original, tagged, np.array([[0., 1.]]))
    np.testing.assert_array_equal(weighted[0, 0], original[0, 0])
    np.testing.assert_array_equal(weighted[0, 1], tagged[0, 0])
    alt, fallback = redistribute(original, tagged, np.zeros((1, 2)))
    np.testing.assert_array_equal(alt, tagged)
    assert diagnostics['applied'] and not fallback['applied']


def run(out):
    started = time.monotonic()
    synthetic_check()
    evidence = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-local-edit-toggle-evidence.json'
    if digest(evidence) != PARENT_REPORT:
        raise ValueError('The frozen local toggle evidence changed')
    frozen = json.loads(evidence.read_text(encoding='utf-8'))
    manifest = PARENT / 'results.json'
    if digest(manifest) != frozen['parent_sha256']:
        raise ValueError('The source cohort changed')
    parent = json.loads(manifest.read_text(encoding='utf-8'))
    if parent['model_sha256'] != frozen['model_sha256'] or parent['strength'] != .8:
        raise ValueError('Unexpected local toggle checkpoint or strength')
    out.mkdir(exist_ok=False)
    entries = {row['pair']: row for row in parent['cases']}
    cases = {}
    for pair, case in frozen['cases'].items():
        def checked(name):
            path = BASE / 'local-edit-toggle-v1' / pair / name
            if digest(path) != case['files'][name]:
                raise ValueError(f'Saved image changed: {pair}/{name}')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError(f'Invalid image geometry or mode: {pair}/{name}')
                return np.array(image)

        humans = tuple(mask_for(entries[pair], photo) for photo in pair.split('-'))
        weights = tuple(edit_weight(mask, 1)[0] if mask.any() else np.zeros(mask.shape) for mask in humans)
        original = checked('target.png')
        outputs = {name: checked(f'{name}-current13.png') for name in ROLES}
        outputs['local'], diagnostics = redistribute(outputs['base'], outputs['local'], weights[0])
        sha = save(out / f'{pair}.png', outputs['local'])
        expected = {name: checked(f'{name}-expected.png') for name in ROLES}
        masks = target_regions(original, humans[0])
        masks.update(edited_core=weights[0] == 1, untouched=weights[0] == 0,
                     neutral_person=masks['neutral_proxy'] & humans[0],
                     neutral_edited=masks['neutral_proxy'] & (weights[0] == 1))
        metrics = compare(outputs, expected, masks)
        gates = gates_for(metrics, outputs, humans, weights)
        cases[pair] = dict(metrics=metrics, gates=gates, diagnostics=diagnostics, sha256=sha)
    elapsed = time.monotonic() - started
    if elapsed >= 60:
        raise TimeoutError('Frozen CPU budget exceeded')
    report = dict(cases=cases, seconds=elapsed, toggle_sha256=PARENT_REPORT, model_sha256=frozen['model_sha256'],
                  source_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'local-chroma-redistribution-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: row['gates'] for name, row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
