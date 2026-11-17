"""Recheck the existing bidirectional PSBench comparison at native resolution."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_psbench_teal_portrait import IDS, verify_source
from probe_reference_chroma_spread import projection
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


def run(folder, out):
    files = {task: verify_source(folder, task) for task in IDS}
    originals = {task: decode_image(folder / f'{task}_start.jpg') for task in IDS}
    engine = SpatialEngine('multiref-v1')
    out.mkdir(exist_ok=False)
    predictions = {}
    for target, reference in (IDS, IDS[::-1]):
        original = originals[target]
        look = decode_image(folder / f'{reference}_end.jpg')
        rendered = engine.prepare(original, look).apply(original)
        path = out / f'{target}.png'
        Image.fromarray(finish_spatial(original, rendered, .8)).save(path)
        predictions[target] = digest(path)
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')

    scores = {}
    for task, original in originals.items():
        truth = decode_image(folder / f'{task}_end.jpg')
        output = decode_image(out / f'{task}.png').astype(np.int16)
        if truth.shape != original.shape or output.shape != original.shape:
            raise ValueError(f'{task}: original, edited and output dimensions differ')
        expected = finish_spatial(original, truth, .8).astype(np.int16)
        delta = expected - original
        response = output - original
        masks = target_regions(original, np.zeros(original.shape[:2], dtype=bool))
        scores[task] = {
            name: dict(pixels=int(mask.sum()), original_mae=float(np.abs(delta[mask]).mean()),
                       result_mae=float(np.abs((output - expected)[mask]).mean()),
                       response_projection=projection(response, delta, mask))
            for name, mask in masks.items() if name in ('whole', 'neutral_proxy') and mask.any()
        }
    first, reverse = (scores[task]['whole'] for task in IDS)
    reproduced = (
        first['original_mae'] - first['result_mae'] >= 1
        and reverse['result_mae'] - reverse['original_mae'] >= 1
        and reverse['response_projection'] < 0
    )
    report = dict(files=files, predictions=predictions, scores=scores, direction_reproduced=reproduced,
                  dimensions_wh={task: list(image.shape[:2][::-1]) for task, image in originals.items()},
                  checkpoint_sha256=engine.checkpoint_sha, strength=.8, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-native-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(scores=scores, direction_reproduced=reproduced), allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.folder, args.out)
