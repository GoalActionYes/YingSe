"""Fit reference before/after changes without using target truth at inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from background_intervention import target_regions
from capacity_audit import block_split, polynomial_features
from independent_coco import digest
from probe_inretouch_public import STYLE, TARGETS
from reference_color_app.media import decode_image
from reference_transfer_probe import crossfit_transfer
from spatial_render import finish_spatial

EVIDENCE = Path(__file__).resolve().parents[2] / 'docs/research/2026-11-18-inretouch-public-pilot-evidence.json'


def run(source, out):
    out.mkdir(exist_ok=False)
    parent = json.loads(EVIDENCE.read_text(encoding='utf-8'))

    def load(entry):
        path = source / entry.removeprefix('example_data/').replace('/', '-')
        assert digest(path) == parent['inputs'][entry]
        return decode_image(path)

    for file, sha256 in parent['predictions'].items():
        assert digest(source / 'run' / file) == sha256
    targets = {name: load(f'example_data/Inference_Data/{name}.jpg') for name in TARGETS}
    sizes = {name: img.shape[0] * img.shape[1] for name, img in targets.items()}
    design = np.concatenate([polynomial_features(img, 2) for img in targets.values()])
    start = time.monotonic()
    predictions = {}
    reference_fits = {}
    ranges = {}
    for style in STYLE:
        before = load(f'example_data/{style}_Before.jpg')
        after = load(f'example_data/{style}_After.jpg')
        assert before.shape == after.shape
        source_design = polynomial_features(before, 2)
        fitted = crossfit_transfer(source_design, after.reshape(-1, 3) / 255, design,
                                   block_split(*before.shape[:2], tile=32))
        crossfit = np.rint(np.clip(fitted['reference_crossfit'], 0, 1) * 255)
        reference_fits[style] = float(np.abs(crossfit-after.reshape(-1, 3)).mean())
        offset = 0
        for name, original in targets.items():
            estimate = fitted['target_prediction'][offset:offset + sizes[name]]
            offset += sizes[name]
            ranges[f'{name}-{style}'] = float(((estimate < 0) | (estimate > 1)).mean())
            quantized = np.rint(np.clip(estimate, 0, 1) * 255).reshape(original.shape).astype(np.uint8)
            path = out / f'{name}-{style}.png'
            Image.fromarray(finish_spatial(original, quantized, parent['strength'])).save(path)
            predictions[path.name] = digest(path)
            assert time.monotonic() - start < 90

    prediction_manifest = out / 'predictions.json'
    prediction_manifest.write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    cases = {}
    for name, original in targets.items():
        truth = load(f'example_data/GT/{name}.jpg')
        assert original.shape == truth.shape
        scaled = finish_spatial(original, truth, parent['strength']).astype(np.int16)
        images = {style: decode_image(out / f'{name}-{style}.png').astype(np.int16) for style in STYLE}
        regions = target_regions(original, np.zeros(original.shape[:2], bool))
        metrics = {}
        for region in ('whole', 'neutral_proxy'):
            mask = regions[region]
            previous = parent['cases'][name][region]
            assert int(mask.sum()) == previous['pixels']
            metrics[region] = dict(output_errors={style: float(np.abs(image[mask]-scaled[mask]).mean())
                                                   for style, image in images.items()},
                                   swap_error=float(np.abs(images[STYLE[0]][mask]-images[STYLE[1]][mask]).mean()),
                                   previous=previous)
        whole = metrics['whole']
        metrics['passed'] = all(whole['output_errors'][style] <
                                min(whole['previous']['outputs_mae'][style], whole['previous']['original_mae'])
                                for style in STYLE) and whole['swap_error'] <= .8 * whole['previous']['reference_swap_mae']
        cases[name] = metrics
    report = dict(input_evidence_sha256=digest(EVIDENCE), script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('inretouch-reference-original-protocol.md')),
                  reference_crossfit_mae=reference_fits, out_of_range_fraction=ranges,
                  predictions=predictions, predictions_sha256=digest(prediction_manifest),
                  cases=cases, passed=all(row['passed'] for row in cases.values()),
                  elapsed_seconds=time.monotonic()-start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
