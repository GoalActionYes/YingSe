"""Cross-check shared exact RGB edits with offline, spatially held-out lookups."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import projection
from reference_color_app.media import decode_image


def color_keys(rgb):
    channels = rgb.astype(np.int32)
    return (channels[..., 0] << 16) | (channels[..., 1] << 8) | channels[..., 2]


def edit_lookup(keys, delta):
    codes, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    offsets = np.cumsum(counts) - counts
    valid = counts >= 4
    lower = offsets[valid] + (counts[valid] - 1) // 2
    upper = offsets[valid] + counts[valid] // 2
    medians = []
    for channel in delta.T:
        ordered = channel[np.lexsort((channel, groups))]
        medians.append((ordered[lower] + ordered[upper]) * .5)
    return codes[valid], np.column_stack(medians)


def score(original, expected, variants, support):
    regions = target_regions(original, np.zeros(support.shape, dtype=bool))
    delta = expected - original
    scores = {}
    for name in ('whole', 'neutral_proxy'):
        region = regions[name]
        measured = support & region
        pixels = int(measured.sum())
        results = {}
        if pixels:
            for method, rgb in variants.items():
                response = rgb.astype(np.int16) - original
                results[method] = dict(mae=float(np.abs((response - delta)[measured]).mean()),
                                       projection=projection(response, delta, measured))
        scores[name] = dict(pixels=pixels, region_pixels=int(region.sum()),
                            coverage=pixels / int(region.sum()), methods=results)
    return scores


def run(folder, out, baseline_evidence):
    with baseline_evidence.open(encoding='utf-8') as f:
        prior = json.load(f)
    files = verify_source(folder, '105')
    if files != prior['files'] or prior['strength'] != .8:
        raise ValueError('Unexpected baseline source or strength')
    source, truth = (decode_image(folder / f'105_{name}.jpg') for name in ('start', 'end'))
    if source.shape != truth.shape or tuple(source.shape[:2]) != (1080, 1440):
        raise ValueError('Unexpected collage dimensions')
    crops = {'portrait': (0, 704), 'tower': (736, 1440)}
    if prior['crop_columns'] != {name: list(crop) for name, crop in crops.items()}:
        raise ValueError('Unexpected baseline crops')
    originals = {name: source[:, start:stop] for name, (start, stop) in crops.items()}
    deltas = {name: np.rint((truth[:, start:stop].astype(np.int16) - originals[name]) * .8).astype(np.int16)
              for name, (start, stop) in crops.items()}
    variants = {}
    support = {}
    for name, original in originals.items():
        path = folder / 'action-pair-v1' / f'{name}.png'
        if digest(path) != prior['predictions'][name]:
            raise ValueError(f'{name}: baseline output hash changed')
        baseline = decode_image(path)
        if baseline.shape != original.shape:
            raise ValueError(f'{name}: baseline geometry changed')
        variants[name] = dict(original=original, current13=baseline,
                              cross_scene=original.copy(), same_scene=original.copy())
        support[name] = np.zeros(original.shape[:2], dtype=bool)
    keys = {name: color_keys(original) for name, original in originals.items()}
    rows, cols = np.ogrid[:1080, :704]
    folds = ((rows // 32 + cols // 32) % 2).astype(bool)
    for test in (folds, ~folds):
        tables = {name: edit_lookup(values[~test], deltas[name][~test]) for name, values in keys.items()}
        codes, portrait_indices, tower_indices = np.intersect1d(
            tables['portrait'][0], tables['tower'][0], assume_unique=True, return_indices=True
        )
        tables = {'portrait': tables['portrait'][1][portrait_indices],
                  'tower': tables['tower'][1][tower_indices]}
        for target, reference in (('portrait', 'tower'), ('tower', 'portrait')):
            measured = test & np.isin(keys[target], codes)
            support[target] |= measured
            positions = np.searchsorted(codes, keys[target][measured])
            base = originals[target][measured]
            for method, origin in (('cross_scene', reference), ('same_scene', target)):
                variants[target][method][measured] = np.clip(
                    np.rint(base + tables[origin][positions]), 0, 255
                ).astype(np.uint8)
    out.mkdir(exist_ok=False)
    artifacts = {}
    for name, results in variants.items():
        for method, value in [('cross_scene', results['cross_scene']), ('same_scene', results['same_scene']),
                              ('support', support[name].astype(np.uint8) * 255)]:
            path = out / f'{name}-{method}.png'
            Image.fromarray(value).save(path)
            artifacts[path.name] = digest(path)
    scores = {name: score(original, original + deltas[name], variants[name], support[name])
              for name, original in originals.items()}
    neutral = [value['neutral_proxy'] for value in scores.values()]
    covered = all(item['coverage'] >= .25 for item in neutral)
    passed = covered and all(
        item['methods']['cross_scene']['mae'] <= min(item['methods'][method]['mae']
                                                    for method in ('original', 'current13')) - 1
        and (item['methods']['cross_scene']['projection'] or 0) > 0
        for item in neutral
    )
    report = dict(scores=scores, artifacts=artifacts, sufficient_neutral_coverage=covered,
                  supported_bidirectional_neutral_transfer=passed, baseline_sha256=digest(baseline_evidence),
                  script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-action-rgb-transfer-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('baseline_evidence', type=Path)
    run(**vars(parser.parse_args()))
