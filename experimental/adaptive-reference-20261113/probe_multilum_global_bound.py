"""Measure the exact RGB8 color-lookup error floor on two consumed light pairs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from background_intervention import target_regions
from capacity_audit import rgb_collision_floor
from independent_coco import digest
from reference_color_app.media import decode_image

EVIDENCE = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-multilum-pilot-evidence.json'


def measure(source, truth, mask):
    scene_values, scene_truth = source[mask], truth[mask]
    common = rgb_collision_floor(scene_values[:, None], scene_truth[:, None])
    width = source.shape[1]
    fractions = []
    for left, right in [(0, width // 2), (width // 2, width)]:
        half = mask[:, left:right]
        pixels = int(half.sum())
        if pixels:
            floor = rgb_collision_floor(source[:, left:right][half][:, None],
                                        truth[:, left:right][half][:, None])
            fractions.append((pixels, floor['minimum_mae']))
    separate = sum(n * error for n, error in fractions) / int(mask.sum())
    assert separate <= common['minimum_mae'] + 1e-10
    return dict(mask_fraction=float(mask.mean()), common_map=common, left_right_minimum_mae=separate,
                identity_mae=float(np.abs(scene_values.astype(np.int16)-scene_truth).mean()))


def run(inputs, output):
    out = output
    out.mkdir(exist_ok=False)
    parent = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    start = time.monotonic()
    cases = {}
    for scene, hashes in parent['inputs'].items():
        images = {}
        for direction in (14, 24):
            path = inputs / f'{scene}-{direction}.jpg'
            assert digest(path) == hashes[str(direction)]
            images[direction] = decode_image(path)
            assert images[direction].shape == (1000, 1500, 3)
        source, truth = images[14], images[24]
        regions = target_regions(source, np.zeros(source.shape[:2], bool))
        cases[scene] = {name: measure(source, truth, regions[name]) for name in ('whole', 'neutral_proxy')}
        assert time.monotonic() - start < 60, 'CPU budget exceeded'
    protocol = Path(__file__).with_name('multilum-global-bound-protocol.md')
    report = dict(inputs_evidence_sha256=digest(EVIDENCE), protocol_sha256=digest(protocol),
                  source_sha256=digest(__file__), scope='paired RGB8 light measurements only',
                  pixel_maps_use_offline_truth=True, case_metrics=cases,
                  near_neutral_bound_ge_5=any(
                      row['neutral_proxy']['common_map']['minimum_mae'] >= 5 for row in cases.values()),
                  seconds=time.monotonic()-start)
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.inputs, args.out)
