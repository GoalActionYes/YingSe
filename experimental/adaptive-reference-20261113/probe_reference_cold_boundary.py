"""Evaluate a reference-only measurement of boundary-aligned color contrast."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import LUMA
from independent_coco import ROOT, digest
from person_segmenter import CHECKPOINT_SHA, PersonSegmenter
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import COOL, ROLES
from probe_reference_chroma_spread import BASE


AXIS = COOL / np.linalg.norm(COOL)


def boundary_measurement(rgb, probability):
    luma = rgb @ LUMA
    chroma = rgb @ AXIS
    samples = []
    for distance in (16, 24):
        for a, b in ((np.s_[:-distance], np.s_[distance:]), (np.s_[:, :-distance], np.s_[:, distance:])):
            matched = np.abs(luma[b] - luma[a]) <= 12
            forward = (probability[a] >= .8) & (probability[b] <= .2) & matched
            backward = (probability[b] >= .8) & (probability[a] <= .2) & matched
            difference = chroma[b] - chroma[a]
            samples.extend((difference[forward], -difference[backward]))
    selected = np.concatenate(samples)
    return (float(np.median(selected)) if len(selected) >= 128 else None), len(selected)


def evaluate(groups, kind, photos):
    results = {}
    for group, versions in groups.items():
        training = [groups[photo] for photo in photos if photo != group
                    and all(row[kind] is not None for row in groups[photo].values())]
        threshold = None
        if len(training) >= 2:
            positive = np.mean([case['local'][kind] for case in training])
            negative = np.mean([case[name][kind] for case in training for name in ('base', 'whole')])
            threshold = float((positive + negative) / 2)
        predictions = {role: (None if threshold is None or item[kind] is None else bool(item[kind] > threshold))
                       for role, item in versions.items()}
        numbers = [versions[name][kind] for name in ROLES]
        order = numbers[1] > max(numbers[0], numbers[2]) if all(value is not None for value in numbers) else None
        results[group] = dict(threshold=threshold, complete_training_scenes=len(training), predictions=predictions,
                              local_above_other_versions=order)
    hits, false_triggers, orders = 0, 0, 0
    for pair, table in results.items():
        is_person = pair in photos
        hits += is_person and table['predictions']['local'] is True
        orders += is_person and table['local_above_other_versions'] is True
        false_triggers += sum(value is True for role, value in table['predictions'].items()
                              if not (is_person and role == 'local'))
    return dict(results=results, true_local=hits, false_triggers=false_triggers, ranked_local=orders,
                abstentions=sum(value is None for row in results.values() for value in row['predictions'].values()),
                passed=hits >= 3 and false_triggers == 0 and orders >= 3)


def synthetic_check():
    picture = np.full((96, 96, 3), 128., dtype=np.float64)
    probability = np.zeros((96, 96))
    probability[:, :48] = 1
    reference, count = boundary_measurement(picture, probability)
    assert reference == 0 and count >= 128
    np.testing.assert_allclose(boundary_measurement(picture + COOL, probability)[0], 0, atol=1e-12)
    picture[:, 48:] += COOL
    np.testing.assert_allclose(boundary_measurement(picture, probability)[0], np.linalg.norm(COOL), atol=1e-12)
    assert boundary_measurement(picture, np.zeros_like(probability)) == (None, 0)


def run(out):
    started = time.monotonic()
    synthetic_check()
    source = Path(__file__).resolve()
    evidence = source.parents[2] / 'docs/research/2026-11-18-local-edit-toggle-evidence.json'
    if digest(evidence) != PARENT_REPORT:
        raise ValueError('Frozen reference cases changed')
    cohort = json.loads(evidence.read_text(encoding='utf-8'))['cases']
    checkpoint = ROOT / '.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    if not checkpoint.is_file() or digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Missing local segmentation checkpoint or hash changed')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    segmenter = PersonSegmenter('cuda')
    out.mkdir(exist_ok=False)
    groups = {}
    for pair, versions in cohort.items():
        features = {}
        for name in ROLES:
            filename = f'{name}-reference.png'
            path = BASE / 'local-edit-toggle-v1' / pair / filename
            if digest(path) != versions['files'][filename]:
                raise ValueError(f'Edited reference changed: {pair}/{filename}')
            with Image.open(path) as image:
                if image.size != (256, 256) or image.mode != 'RGB':
                    raise ValueError('Invalid reference size or mode')
                rgb = np.array(image)
            probability, stats = segmenter.predict(rgb)
            boundary, count = boundary_measurement(rgb, probability)
            mask_path = out / f'{pair}-{name}.npy'
            np.save(mask_path, probability, allow_pickle=False)
            features[name] = dict(boundary=boundary, pairs=count, whole=float(rgb.mean(axis=(0, 1)) @ AXIS),
                                  person_fraction=stats['fraction'], probability_sha256=digest(mask_path),
                                  reference_sha256=versions['files'][filename])
        groups[pair] = features
    artifact = out / 'features.json'
    artifact.write_text(json.dumps(groups, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    photos = [pair for pair, case in cohort.items() if case['gates']['positive']]
    if len(photos) != 4 or len(cohort) != 6:
        raise ValueError('Unexpected cohort size')
    comparisons = {kind: evaluate(groups, kind, photos) for kind in ('boundary', 'whole')}
    seconds = time.monotonic() - started
    if seconds >= 60:
        raise TimeoutError('Fixed reference screening budget exceeded')
    report = dict(groups=groups, comparisons=comparisons, seconds=seconds, source_sha256=digest(source),
                  features_sha256=digest(artifact), segmenter_sha256=CHECKPOINT_SHA, toggle_sha256=PARENT_REPORT,
                  protocol_sha256=digest(source.with_name('reference-cold-boundary-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: {name: v for name, v in row.items() if name != 'results'}
                      for k, row in comparisons.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
