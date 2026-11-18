"""Measure cross-scene transfer when a reference's pre-grade version is known."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_s24_calibrated_pilot import SpatialEngine, decode_image, digest, finish_spatial, image, region_scores, target_regions


def features(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.float32) / 255, -1, 0)
    return np.stack((np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b), axis=-1)


def matrix(source, graded):
    x = features(source[::8, ::8]).reshape(-1, 10)
    y = (graded[::8, ::8].astype(np.float32) - source[::8, ::8]).reshape(-1, 3) / 255
    return np.linalg.solve(x.T @ x / len(x) + 0.001 * np.eye(10), x.T @ y / len(x)).astype(np.float32)


def render(source, transform):
    result = np.empty_like(source)
    for start in range(0, len(source), 128):
        tile = source[start:start + 128]
        result[start:start + len(tile)] = np.rint(np.clip(tile + 255 * (features(tile) @ transform), 0, 255)).astype(np.uint8)
    return finish_spatial(source, result, .8)


def run(assets, evidence_path, out):
    torch.set_num_threads(4)
    prior = json.loads(evidence_path.read_bytes())
    assert digest(assets / 'assets.json') == prior['assets_sha256']
    assert digest(assets / 'plan.json') == prior['selection_sha256']
    manifest = json.loads((assets / 'assets.json').read_bytes())
    excluded = [key for key, row in prior['cases'].items() if not all(g['passed'] for g in row['geometry'])]
    ids = [key for key in prior['cases'] if key not in excluded]
    assert len(ids) == 4
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == prior['weight_sha256']
    out.mkdir()
    predictions, cases = {}, {}

    def load(scene, grade):
        return decode_image(image(assets, manifest, scene, f'srgb_images_style_{grade}'))

    for index, key in enumerate(ids):
        source = load(key, 1)
        near, far = ids[(index + 1) % 4], ids[(index + 2) % 4]
        checks = dict(references=[(near, 2), (far, 2), (near, 3)])
        near_raw = load(near, 1)
        identity = render(source, matrix(near_raw, near_raw))
        checks['identity'] = bool(np.array_equal(identity, source))
        predictions[key] = {}
        for slot, (donor, style) in enumerate(checks['references']):
            pair_raw, pair_grade = load(donor, 1), load(donor, style)
            assert pair_raw.shape == pair_grade.shape
            frames = {
                'paired': render(source, matrix(pair_raw, pair_grade)),
                'current': finish_spatial(source, engine.prepare(source, pair_grade).apply(source), .8),
            }
            for name, frame in frames.items():
                path = out / f'{key}-{name}-{slot}.png'
                Image.fromarray(frame).save(path)
                predictions[key][path.name] = digest(path)
        cases[key] = checks
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2) + '\n', encoding='utf-8')

    for key in ids:
        source = load(key, 1)
        desired = [finish_spatial(source, load(key, grade), .8) for grade in (2, 3)]
        regions = target_regions(source, np.zeros(source.shape[:2], dtype=bool))
        for label in ('whole', 'neutral_proxy'):
            comparable = {
                name: [decode_image(out / f'{key}-{name}-{slot}.png') for slot in range(3)]
                for name in ('paired', 'current')
            }
            comparable['unchanged'] = [source] * 3
            cases[key][label] = {name: region_scores(frames, desired, regions[label])
                                 for name, frames in comparable.items()}
        scores = cases[key]['whole']['paired']
        cases[key]['joint'] = (scores['expected_delta'] >= 1 and scores['same_swap'] < min(scores['different_swap'])
                               and scores['projection'] is not None and scores['projection'] >= .5)
    summary = {}
    for label in ('whole', 'neutral_proxy'):
        groups = {method: np.array([row[label][method]['errors'] for row in cases.values()])
                  for method in ('paired', 'current', 'unchanged')}
        summary[label] = {method: float(values.mean()) for method, values in groups.items()}
        summary[label]['worst_regression'] = float((groups['paired'] - groups['current']).max())
    joint = sum(row['joint'] for row in cases.values())
    passed = joint >= 3 and all(row['identity'] for row in cases.values()) and all(
        s['paired'] <= .9 * s['current'] and s['paired'] < s['unchanged'] and s['worst_regression'] <= 2
        for s in summary.values()
    )
    report = dict(excluded=excluded, joint=joint, passed=passed, summary=summary, cases=cases,
                  prior_sha256=digest(evidence_path), assets_sha256=prior['assets_sha256'],
                  weight_sha256=engine.checkpoint_sha, source_sha256=digest(Path(__file__)), predictions=predictions)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(joint=joint, passed=passed, summary=summary)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('evidence_path', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
