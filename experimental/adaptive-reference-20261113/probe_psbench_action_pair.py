"""Render the two scenes shown separately in the PSBench 105 action tutorial."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_psbench_teal_portrait import verify_source
from probe_reference_chroma_spread import projection
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


def run(folder, out):
    files = verify_source(folder, '105')
    original = decode_image(folder / '105_start.jpg')
    edited = decode_image(folder / '105_end.jpg')
    if original.shape != edited.shape or tuple(original.shape[:2]) != (1080, 1440):
        raise ValueError('Unexpected collage dimensions')
    crops = {'portrait': slice(0, 704), 'tower': slice(736, 1440)}
    images = {name: (original[:, crop], edited[:, crop]) for name, crop in crops.items()}
    geometry = {}
    for name, pair in images.items():
        small = [np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR)) for rgb in pair]
        geometry[name] = structure_score(*small)
    if not all(item['passed'] for item in geometry.values()):
        raise ValueError(f'Unexpected change in before/after geometry: {geometry}')

    engine = SpatialEngine('multiref-v1')
    out.mkdir(exist_ok=False)
    predictions = {}
    for target, reference in [('portrait', 'tower'), ('tower', 'portrait')]:
        source = images[target][0]
        look = images[reference][1]
        full = engine.prepare(source, look).apply(source)
        path = out / f'{target}.png'
        Image.fromarray(finish_spatial(source, full, .8)).save(path)
        predictions[target] = digest(path)
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')

    scores = {}
    for name, (source, truth) in images.items():
        output = decode_image(out / f'{name}.png').astype(np.int16)
        expected = finish_spatial(source, truth, .8).astype(np.int16)
        delta = expected - source
        response = output - source
        masks = target_regions(source, np.zeros(source.shape[:2], dtype=bool))
        scores[name] = {
            region: dict(pixels=int(mask.sum()), original_mae=float(np.abs(delta[mask]).mean()),
                         result_mae=float(np.abs((output - expected)[mask]).mean()),
                         response_projection=projection(response, delta, mask))
            for region, mask in masks.items() if region in ('whole', 'neutral_proxy') and mask.any()
        }
    passed = all(
        row['whole']['original_mae'] - row['whole']['result_mae'] >= 1
        and row['whole']['response_projection'] > 0
        for row in scores.values()
    )
    report = dict(files=files, geometry=geometry, predictions=predictions, scores=scores,
                  bidirectional_improvement=passed, crop_columns={name: [part.start, part.stop]
                                                                 for name, part in crops.items()},
                  strength=.8, checkpoint_sha256=engine.checkpoint_sha, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-action-pair-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(scores=scores, bidirectional_improvement=passed), allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.folder, args.out)
