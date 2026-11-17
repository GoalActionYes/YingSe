"""Run the frozen ramp screen on different source photos and edit boundaries."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from independent_coco import ROOT, digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import COOL, mask_for, save
from probe_reference_chroma_spread import BASE, PARENT
from probe_reference_cold_ramp import ramp_measurement


RAMP_REPORT = '5f0213ab462bc8078a642c69d15ec1b39c476108cbc3e0d40bf871e4897bf184'
THRESHOLD = 5.844976007785105
LOCAL_ROLES = ('hard', 'wide')


def local_styles(original, person):
    base = 127.5 + .88 * (original.astype(float) - 127.5) + np.array([6., 2., -5.])
    if person.any():
        hard = (~person).astype(float)
        wide = np.zeros(person.shape)
        expansion = Image.fromarray(person.astype(np.uint8))
        for _ in range(32):
            wide += np.asarray(expansion) == 0
            expansion = expansion.filter(ImageFilter.MaxFilter(3))
        wide /= 32
    else:
        hard = wide = np.zeros(person.shape)
    rows = dict(base=base, whole=base + COOL, hard=base + hard[..., None] * COOL,
                wide=base + wide[..., None] * COOL)
    images = {name: np.rint(pixels.clip(0, 255)).astype(np.uint8) for name, pixels in rows.items()}
    for name in LOCAL_ROLES:
        if not np.array_equal(images[name][person], images['base'][person]):
            raise ValueError('Modified a protected pixel')
    return images


def run(out):
    started = time.monotonic()
    source = Path(__file__).resolve()
    docs = source.parents[2] / 'docs/research'
    toggle_path = docs / '2026-11-18-local-edit-toggle-evidence.json'
    ramp_path = docs / '2026-11-18-reference-cold-ramp-evidence.json'
    if digest(toggle_path) != PARENT_REPORT or digest(ramp_path) != RAMP_REPORT:
        raise ValueError('Frozen evidence changed')
    toggle, ramp = [json.loads(path.read_text(encoding='utf-8')) for path in (toggle_path, ramp_path)]
    if (any(row['threshold'] != THRESHOLD for pair, row in ramp['comparison']['results'].items()
            if ramp['groups'][pair]['base']['ramp'] is None)
            or not ramp['comparison']['passed']):
        raise ValueError('Frozen threshold does not match')
    if digest(PARENT / 'results.json') != toggle['parent_sha256']:
        raise ValueError('Unexpected developer cohort')
    data = json.loads((PARENT / 'results.json').read_text(encoding='utf-8'))
    target_ids, previous_refs = zip(*(pair.split('-') for pair in toggle['cases']))
    if len(set(target_ids)) != 6 or set(target_ids) & set(previous_refs):
        raise ValueError('Unexpected original-content split')
    checkpoint = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    if not checkpoint.is_file() or digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Missing frozen automatic segmentation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    segmenter = PersonSegmenter('cuda')
    out.mkdir(exist_ok=False)
    cases, hits, false_triggers = {}, dict(hard=0, wide=0), 0
    for case in data['cases']:
        pair = case['pair']
        photo, _ = pair.split('-')
        path = BASE / 'local-edit-toggle-v1' / pair / 'target.png'
        files = toggle['cases'][pair]['files']
        if digest(path) != files[path.name]:
            raise ValueError('Frozen original image changed')
        with Image.open(path) as image:
            pixels = np.array(image)
        human = mask_for(case, photo)
        present = bool(human.any())
        options = local_styles(pixels, human)
        versions = {}
        for name, rgb in options.items():
            if name in ('base', 'whole'):
                previous = BASE / 'local-edit-toggle-v1' / pair / f'{name}-target-full.png'
                with Image.open(previous) as im:
                    if digest(previous) != files[previous.name] or not np.array_equal(rgb, np.asarray(im)):
                        raise ValueError('Frozen old control did not reproduce')
            rgb_hash = save(out / f'{photo}-{name}.png', rgb)
            probability, _ = segmenter.predict(rgb)
            probability_path = out / f'{photo}-{name}.npy'
            np.save(probability_path, probability, allow_pickle=False)
            score, count = ramp_measurement(rgb, probability)
            predicted = None if score is None else score > THRESHOLD
            expected = present and name in LOCAL_ROLES
            if predicted is True:
                if expected:
                    hits[name] += 1
                else:
                    false_triggers += 1
            versions[name] = dict(score=score, pairs=count, predicted_local=predicted, expected_local=expected,
                                  rgb_sha256=rgb_hash, probability_sha256=digest(probability_path))
        cases[photo] = dict(person_present=present, source_sha256=files[path.name], versions=versions)
    elapsed = time.monotonic() - started
    if elapsed >= 60:
        raise TimeoutError('Frozen robustness budget exceeded')
    report = dict(cases=cases, hits=hits, false_triggers=false_triggers,
                  passed=min(hits.values()) >= 3 and false_triggers == 0, threshold=THRESHOLD, seconds=elapsed,
                  ramp_sha256=RAMP_REPORT, toggle_sha256=PARENT_REPORT, segmenter_sha256=CHECKPOINT_SHA,
                  source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'reference-cold-ramp-transfer-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(hits=hits, false_triggers=false_triggers, passed=report['passed'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
