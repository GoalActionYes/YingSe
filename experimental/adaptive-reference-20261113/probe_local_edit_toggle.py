"""Measure 13's incremental response to a controlled local color grade."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import LUMA, edit_weight, region_delta, target_regions
from independent_coco import digest
from probe_gray_edge_correction import PARENT_SHA
from probe_reference_chroma_spread import BASE, PARENT, projection, read
from spatial_render import SpatialEngine, finish_spatial
from spatial_store import SpatialStore


ROLES = ('base', 'local', 'whole')
COOL = np.array([-8., (8 * .299 - 17 * .114) / .587, 17.])


def versions(rgb, human):
    base = 127.5 + .88 * (rgb.astype(np.float64) - 127.5) + np.array([6., 2., -5.])
    weight = edit_weight(human, 1)[0] if human.any() else np.zeros(human.shape)
    raw = dict(base=base, local=base + weight[..., None] * COOL, whole=base + COOL)
    images = {name: np.rint(value.clip(0, 255)).astype(np.uint8) for name, value in raw.items()}
    if (images['local'][weight == 0] != images['base'][weight == 0]).any():
        raise ValueError('The local edit changed protected pixels')
    return images, weight, {name: float(((value < 0) | (value > 255)).mean()) for name, value in raw.items()}


def save(path, image):
    Image.fromarray(image).save(path)
    return digest(path)


def mask_for(case, image_id):
    name = f'{image_id}-human-mask.png'
    path = BASE / 'independent-coco' / name
    if digest(path) != case['parents'][name]:
        raise ValueError('Authoring mask changed')
    with Image.open(path) as image:
        return np.asarray(image.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0


def compare(images, authored, masks):
    results = {}
    for name in ('local', 'whole'):
        delta = images[name].astype(np.int16) - images['base']
        truth = authored[name].astype(np.int16) - authored['base']
        error = np.abs(delta - truth)
        results[name] = {}
        for region, mask in masks.items():
            if mask.any():
                results[name][region] = dict(
                    observed=region_delta(images[name], images['base'], mask),
                    expected=region_delta(authored[name], authored['base'], mask),
                    error=float(error[mask].mean()),
                    projection=projection(delta, truth, mask),
                )
            else:
                results[name][region] = None
    return results


def gates_for(metrics, images, humans, weights):
    positive = all(int((weight == 1).sum()) > 256 for weight in weights) and all(
        human.any() for human in humans
    )
    local = metrics['local']
    checks = {}
    if positive:
        checks['local_background_direction'] = .75 <= local['edited_core']['projection'] <= 1.25
        checks['local_background_error'] = local['edited_core']['error'] <= 3
        checks['local_person_protection'] = local['person']['error'] <= 1.5
        neutral = local['neutral_person']
        checks['local_neutral_person'] = (neutral['error'] <= 1.5
                                         if neutral and neutral['observed']['pixels'] >= 64 else None)
    checks['whole_response'] = .75 <= metrics['whole']['whole']['projection'] <= 1.25
    checks['no_person'] = (bool(np.array_equal(images['base'], images['local']))
                           if not any(human.any() for human in humans) else None)
    return dict(positive=positive, checks=checks)


def run(out):
    started = time.monotonic()
    np.testing.assert_allclose(COOL @ LUMA, 0, atol=1e-12)
    torch.set_num_threads(4)
    manifest = PARENT / 'results.json'
    if digest(manifest) != PARENT_SHA:
        raise ValueError('Frozen development cohort changed')
    parent = json.loads(manifest.read_text(encoding='utf-8'))
    engine = SpatialEngine('multiref-v1')
    if parent['model_sha256'] != engine.checkpoint_sha or parent['strength'] != .8:
        raise ValueError('Frozen model or strength changed')
    out.mkdir(exist_ok=False)
    cases, pending = {}, {}
    for case in parent['cases']:
        pair = case['pair']
        target_id, ref_id = pair.split('-')
        human, ref_human = [mask_for(case, photo) for photo in (target_id, ref_id)]
        target = read(case['input_file'])
        identity = next(v for v in case['variants'] if v['name'] == 'identity')
        ref = read(identity['reference'])
        authored, weight, target_clip = versions(target, human)
        references, ref_weight, ref_clip = versions(ref, ref_human)
        expected = {name: finish_spatial(target, value, .8) for name, value in authored.items()}
        folder = out / pair
        folder.mkdir()
        target_path = folder / 'target.png'
        files = {target_path.name: save(target_path, target)}
        files['reference.png'] = save(folder / 'reference.png', ref)
        for name in ROLES:
            files[f'{name}-reference.png'] = save(folder / f'{name}-reference.png', references[name])
            files[f'{name}-expected.png'] = save(folder / f'{name}-expected.png', expected[name])
            files[f'{name}-target-full.png'] = save(folder / f'{name}-target-full.png', authored[name])
        store = SpatialStore(folder / 'store', engine)
        target_media = store.add_media(target_path)
        outputs = {}
        for name in ROLES:
            reference_media = store.add_media(folder / f'{name}-reference.png')
            mapping = store.analyze(target_media['id'], reference_media['id'])
            if mapping['diagnostics']['certified_aligned']['accepted']:
                raise ValueError('Expected cross-scene inference')
            outputs[name] = store.render(mapping['id'], target, .8)
            files[f'{name}-current13.png'] = save(folder / f'{name}-current13.png', outputs[name])
        cases[pair] = dict(files=files, target_clip=target_clip, reference_clip=ref_clip,
                           target_person=int(human.sum()), reference_person=int(ref_human.sum()),
                           target_edited_core=int((weight == 1).sum()), reference_edited_core=int((ref_weight == 1).sum()))
        masks = target_regions(target, human)
        masks.update(edited_core=weight == 1, untouched=weight == 0,
                     neutral_person=masks['neutral_proxy'] & human, neutral_edited=masks['neutral_proxy'] & (weight == 1))
        pending[pair] = (outputs, expected, masks, (human, ref_human), (weight, ref_weight))
    for pair, (images, expected, masks, humans, weights) in pending.items():
        results = compare(images, expected, masks)
        cases[pair].update(metrics=results, gates=gates_for(results, images, humans, weights))
    elapsed = time.monotonic() - started
    if elapsed >= 120:
        raise TimeoutError('Frozen render budget exceeded')
    report = dict(cases=cases, seconds=elapsed, parent_sha256=PARENT_SHA, model_sha256=engine.checkpoint_sha,
                  source_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'local-edit-toggle-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({pair: row['gates'] for pair, row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
