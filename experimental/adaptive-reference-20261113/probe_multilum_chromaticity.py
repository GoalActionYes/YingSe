"""Measure chromaticity and brightness separately on the saved lighting pairs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from background_intervention import target_regions
from independent_coco import digest
from reference_color_app.media import decode_image

EVIDENCE = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-multilum-pilot-evidence.json'


def linear_rgb(rgb):
    encoded = rgb.astype(np.float64) / 255
    return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055) / 1.055) ** 2.4)


def differences(before, after):
    a, b = linear_rgb(before), linear_rgb(after)
    floor = 1 / (255 * 12.92)
    channel_change = np.log(np.maximum(b, floor)) - np.log(np.maximum(a, floor))
    chromaticity = channel_change[:, [0, 2]] - channel_change[:, 1:2]
    weights = np.array([.2126, .7152, .0722])
    lightness = np.log(np.maximum(b @ weights, floor)) - np.log(np.maximum(a @ weights, floor))
    return chromaticity, lightness


def measure(original, actual, old, new, region, valid):
    mask = region & valid
    if not mask.any():
        return {'retained_fraction': 0.0, 'usable': False}
    truth, light = differences(original[mask], actual[mask])
    estimated, estimated_light = differences(old[mask], new[mask])
    truth_norm = float(np.sum(truth ** 2))
    observed = float(np.sqrt(np.mean(truth ** 2)))
    coverage = float(mask.sum() / region.sum())
    output_extreme = ((old[mask] < 16) | (old[mask] > 239) |
                      (new[mask] < 16) | (new[mask] > 239)).any(axis=1)
    return dict(retained_fraction=coverage, output_extreme_fraction=float(output_extreme.mean()),
                observed_chroma_rms=observed, generated_chroma_rms=float(np.sqrt(np.mean(estimated ** 2))),
                chroma_error_rms=float(np.sqrt(np.mean((truth - estimated) ** 2))),
                chroma_projection=float(np.sum(estimated * truth) / truth_norm) if truth_norm else None,
                observed_log_luminance_rms=float(np.sqrt(np.mean(light ** 2))),
                generated_log_luminance_rms=float(np.sqrt(np.mean(estimated_light ** 2))),
                usable=coverage >= .5 and observed >= .05)


def run(inputs, out):
    out.mkdir(exist_ok=False)
    parent = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    cases = {}
    start = time.monotonic()
    for scene, hashes in parent['inputs'].items():
        source = {}
        generated = {}
        for direction in ('14', '24'):
            jpg = inputs / f'{scene}-{direction}.jpg'
            png = inputs / f'{scene}-output-ref{direction}.png'
            assert digest(jpg) == hashes[direction]
            assert digest(png) == parent['outputs'][scene][direction]
            source[direction], generated[direction] = decode_image(jpg), decode_image(png)
            assert source[direction].shape == generated[direction].shape == (1000, 1500, 3)
        original, actual = source['14'], source['24']
        regions = target_regions(original, np.zeros(original.shape[:2], dtype=bool))
        valid = ((original >= 16) & (original <= 239) & (actual >= 16) & (actual <= 239)).all(axis=2)
        cases[scene] = {name: measure(original, actual, generated['14'], generated['24'], regions[name], valid)
                        for name in ('whole', 'neutral_proxy')}
        assert time.monotonic() - start < 60
    protocol = Path(__file__).with_name('multilum-chromaticity-protocol.md')
    report = dict(inputs_evidence_sha256=digest(EVIDENCE), script_sha256=digest(__file__),
                  protocol_sha256=digest(protocol), cases=cases,
                  supports_chroma_diagnostic=all(case['neutral_proxy']['usable'] for case in cases.values()),
                  seconds=time.monotonic()-start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.inputs, args.out)
