"""Measure the cold-color gradient along matched background-only samples."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter

from background_intervention import LUMA
from independent_coco import digest
from probe_local_edit_toggle import COOL
from probe_reference_cold_boundary import AXIS, evaluate
from probe_reference_chroma_spread import BASE


BOUNDARY_SHA = '55b1a551b3b63039f2bac086bf45a1426fdd89c629855cf298499a68ce535769'


def boundary_distance(person):
    expanded = Image.fromarray(person.astype(np.uint8) * 255)
    distance = np.zeros(person.shape, dtype=np.uint8)
    for radius in range(27):
        distance += np.asarray(expanded) == 0
        if radius < 26:
            expanded = expanded.filter(ImageFilter.MaxFilter(3))
    return distance


def ramp_measurement(rgb, probability):
    person = probability >= .5
    if not person.any():
        return None, 0
    distance = boundary_distance(person)
    background = probability <= .2
    luma = rgb @ LUMA
    chroma = rgb @ AXIS
    samples = []
    for a, mid, b in ((np.s_[:-16], np.s_[8:-8], np.s_[16:]), (np.s_[16:], np.s_[8:-8], np.s_[:-16]),
                      (np.s_[:, :-16], np.s_[:, 8:-8], np.s_[:, 16:]),
                      (np.s_[:, 16:], np.s_[:, 8:-8], np.s_[:, :-16])):
        source_distance = distance[a]
        valid = ((source_distance >= 2) & (source_distance <= 10) & (distance[mid] > source_distance)
                 & (distance[b] > distance[mid]) & background[a] & background[mid] & background[b]
                 & (np.abs(luma[b] - luma[a]) <= 12))
        samples.append((chroma[b] - chroma[a])[valid])
    combined = np.concatenate(samples)
    return (float(np.median(combined)) if len(combined) >= 128 else None), len(combined)


def synthetic_check():
    rgb = np.full((96, 96, 3), 128.)
    probability = np.zeros((96, 96))
    probability[:, :48] = 1
    base, count = ramp_measurement(rgb, probability)
    assert base == 0 and count >= 128
    np.testing.assert_allclose(ramp_measurement(rgb + COOL, probability)[0], 0, atol=1e-12)
    weight = ((np.arange(96) - 47) / 32).clip(0, 1)
    altered = rgb + weight[None, :, None] * COOL
    np.testing.assert_allclose(ramp_measurement(altered, probability)[0], np.linalg.norm(COOL) / 2, atol=1e-12)
    assert ramp_measurement(rgb, np.zeros_like(probability)) == (None, 0)


def run(out):
    started = time.monotonic()
    synthetic_check()
    source = Path(__file__).resolve()
    evidence = source.parents[2] / 'docs/research/2026-11-18-reference-cold-boundary-evidence.json'
    if digest(evidence) != BOUNDARY_SHA:
        raise ValueError('The frozen reference analysis changed')
    prior = json.loads(evidence.read_text(encoding='utf-8'))
    toggle_path = source.parents[2] / 'docs/research/2026-11-18-local-edit-toggle-evidence.json'
    if digest(toggle_path) != prior['toggle_sha256']:
        raise ValueError('The frozen reference cases changed')
    toggle = json.loads(toggle_path.read_text(encoding='utf-8'))
    photos = [name for name, item in toggle['cases'].items() if item['gates']['positive']]
    out.mkdir(exist_ok=False)
    groups = {}
    for case, entries in prior['groups'].items():
        versions = {}
        for role, row in entries.items():
            path = BASE / 'local-edit-toggle-v1' / case / f'{role}-reference.png'
            probability_path = BASE / 'reference-cold-boundary-v1' / f'{case}-{role}.npy'
            if digest(path) != row['reference_sha256'] or digest(probability_path) != row['probability_sha256']:
                raise ValueError(f'Frozen RGB or automatic probability changed: {case}/{role}')
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (256, 256):
                    raise ValueError('Unexpected reference geometry')
                rgb = np.array(image)
            probability = np.load(probability_path, allow_pickle=False)
            if probability.shape != rgb.shape[:2] or not np.isfinite(probability).all():
                raise ValueError('Unexpected automatic probability')
            ramp, count = ramp_measurement(rgb, probability)
            versions[role] = dict(ramp=ramp, pairs=count, prior_boundary=row['boundary'])
        groups[case] = versions
    decision = evaluate(groups, 'ramp', photos)
    seconds = time.monotonic() - started
    if seconds >= 30:
        raise TimeoutError('Frozen CPU budget exceeded')
    report = dict(groups=groups, comparison=decision, baseline=prior['comparisons']['boundary'], seconds=seconds,
                  boundary_sha256=BOUNDARY_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('reference-cold-ramp-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({name: value for name, value in decision.items() if name != 'results'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
