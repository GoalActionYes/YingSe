"""Balance subject/background neutral chroma in a frozen two-image grade."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import LUMA, edit_weight, target_regions
from independent_coco import digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_gray_edge_correction import authored
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import ROLES, compare, gates_for, mask_for, save
from probe_reference_chroma_spread import BASE, PARENT, ROOT


PSBENCH_SHA = '8e3f455399879e457af2fe5444e28806affa5114f29067a2087324584ad410b6'


def relation(rgb_support, rgb_measured, person):
    origin = rgb_support.astype(float)
    intensity = origin @ LUMA
    low_chroma = np.ptp(origin, axis=-1) <= 36
    values = rgb_measured.astype(float)
    chroma = values - (values @ LUMA)[..., None]
    count = 0
    total = np.zeros(3)
    for low in range(40, 220, 30):
        select = low_chroma & (intensity >= low) & (intensity < low + 30)
        foreground = select & (person >= .8)
        background = select & (person <= .2)
        weight = int(min(np.count_nonzero(foreground), np.count_nonzero(background)))
        if weight >= 64:
            count += weight
            total += weight * (chroma[foreground].mean(axis=0) - chroma[background].mean(axis=0))
    return (total / count if count >= 256 else None), count


def balanced_render(baseline, person, difference):
    vector = .5 * difference
    maximum = np.max(np.abs(vector))
    if maximum > 8:
        vector = vector * (8 / maximum)
    vector -= vector @ LUMA
    probability = person.astype(np.float64)
    change = (probability - probability.mean())[..., None] * vector
    np.testing.assert_allclose(change.mean(axis=(0, 1)), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(change @ LUMA, 0, atol=1e-12, rtol=0)
    raw = baseline + change
    rendered = np.rint(raw.clip(0, 255)).astype(np.uint8)
    return rendered, dict(vector=vector.tolist(), clip_fraction=float(((raw < 0) | (raw > 255)).mean()),
                          mean_luminance_error=float(np.abs((rendered.astype(float) - baseline) @ LUMA).mean()))


class BalanceProbe:
    def __init__(self):
        self.segmenter = PersonSegmenter('cuda')
        self.target_masks = {}

    def measure(self, original, reference, baseline):
        if np.array_equal(original, reference):
            return original.copy(), dict(applied=False, reason='same_image')
        key = (original.shape, original.tobytes())
        if key not in self.target_masks:
            self.target_masks[key] = self.segmenter.predict(original)[0]
        target_mask = self.target_masks[key]
        measured, target_support = relation(original, baseline, target_mask)
        debug = dict(applied=False, target_support=target_support)
        if target_mask.mean() < .01 or measured is None:
            return baseline.copy(), dict(**debug, reason='target_support')
        reference_mask, _ = self.segmenter.predict(reference)
        expected, ref_support = relation(reference, reference, reference_mask)
        debug['reference_support'] = ref_support
        if expected is None:
            return baseline.copy(), dict(**debug, reason='reference_support')
        output, stats = balanced_render(baseline, target_mask, expected - measured)
        debug.update(applied=True, observed=measured.tolist(), desired=expected.tolist(), **stats)
        return output, debug


def checked_image(path, sha):
    if digest(path) != sha:
        raise ValueError(f'Frozen input changed: {path.name}')
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError('Expected RGB8')
        return np.array(image)


def controlled(probe, out, frozen, parent):
    cases = {}
    for pair, row in frozen['cases'].items():
        folder = BASE / 'local-edit-toggle-v1' / pair
        target = checked_image(folder / 'target.png', row['files']['target.png'])
        records, outputs, expected = {}, {}, {}
        for name in ROLES:
            files = row['files']
            ref = checked_image(folder / f'{name}-reference.png', files[f'{name}-reference.png'])
            baseline = checked_image(folder / f'{name}-current13.png', files[f'{name}-current13.png'])
            outputs[name], stats = probe.measure(target, ref, baseline)
            records[name] = dict(**stats, sha256=save(out / f'{pair}-{name}.png', outputs[name]))
        for name in ROLES:
            expected[name] = checked_image(folder / f'{name}-expected.png', row['files'][f'{name}-expected.png'])
        humans = tuple(mask_for(parent[pair], photo) for photo in pair.split('-'))
        weights = tuple(edit_weight(mask, 1)[0] if mask.any() else np.zeros(mask.shape) for mask in humans)
        scopes = target_regions(target, humans[0])
        scopes.update(edited_core=weights[0] == 1, untouched=weights[0] == 0,
                      neutral_person=humans[0] & scopes['neutral_proxy'],
                      neutral_edited=(weights[0] == 1) & scopes['neutral_proxy'])
        metrics = compare(outputs, expected, scopes)
        cases[pair] = dict(output=records, metrics=metrics, gates=gates_for(metrics, outputs, humans, weights))
    return cases


def run(out):
    start = time.monotonic()
    docs = ROOT / 'docs/research'
    toggle = docs / '2026-11-18-local-edit-toggle-evidence.json'
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    if digest(toggle) != PARENT_REPORT or digest(pair) != PSBENCH_SHA:
        raise ValueError('Frozen development protocols changed')
    frozen = json.loads(toggle.read_text(encoding='utf-8'))
    manifest = PARENT / 'results.json'
    if digest(manifest) != frozen['parent_sha256']:
        raise ValueError('Unexpected source group')
    parent = {p['pair']: p for p in json.loads(manifest.read_text(encoding='utf-8'))['cases']}
    checkpoint = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    if not checkpoint.is_file() or digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Missing frozen automatic segmentation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(exist_ok=False)
    probe = BalanceProbe()
    sample = np.tile(np.array([120, 124, 130], dtype=np.uint8), (16, 16, 1))
    same, status = probe.measure(sample, sample.copy(), 255 - sample)
    assert np.array_equal(sample, same) and status['reason'] == 'same_image'
    cases = controlled(probe, out, frozen, parent)
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    seconds = time.monotonic() - start
    if seconds >= 90:
        raise TimeoutError('Frozen balance screening budget exceeded')
    report = dict(cases=cases, authored=real, seconds=seconds, same_image_held=True,
                  segmenter_sha256=CHECKPOINT_SHA, toggle_sha256=PARENT_REPORT, psbench_sha256=PSBENCH_SHA,
                  source_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'neutral-relation-balance-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(controlled={key: row['gates'] for key, row in cases.items()},
                          authored={key: row['gate'] for key, row in real.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
