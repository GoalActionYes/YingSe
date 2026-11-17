"""Check two-image reference behavior on the four public INRetouch example pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial

STRENGTH = .8
TARGETS = ('a0006-IMG_2787', 'a0007-IMG_2480')
STYLE = ('Example1', 'Example2')
TREE = '1f1269732b9b58beb9cb2c8e05e09d3223793406'


def checked_image(members, directory, entry):
    path = directory / entry.removeprefix('example_data/').replace('/', '-')
    data = path.read_bytes()
    assert len(data) == members[entry]['size']
    header = f'blob {len(data)}\0'.encode()
    assert hashlib.sha1(header + data).hexdigest() == members[entry]['sha']
    return decode_image(path), digest(path)


def run(source, tree_path, out):
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    assert tree['sha'] == TREE and not tree['truncated']
    members = {item['path']: item for item in tree['tree']}
    out.mkdir(exist_ok=False)
    hashes = {}

    def load(name):
        image, hashes[name] = checked_image(members, source, name)
        return image

    inputs = {target: load(f'example_data/Inference_Data/{target}.jpg') for target in TARGETS}
    styles = {style: load(f'example_data/{style}_After.jpg') for style in STYLE}
    engine = SpatialEngine('multiref-v1')
    start = time.monotonic()
    predictions = {}
    for target, original in inputs.items():
        for style, reference in styles.items():
            fitted = engine.prepare(original, reference)
            output = finish_spatial(original, fitted.apply(original), STRENGTH)
            path = out / f'{target}-{style}.png'
            Image.fromarray(output).save(path)
            predictions[path.name] = digest(path)
            assert time.monotonic() - start < 90
    elapsed = time.monotonic() - start
    prediction_manifest = out / 'predictions.json'
    prediction_manifest.write_text(json.dumps(predictions, indent=2), encoding='utf-8')

    for style, edited in styles.items():
        assert load(f'example_data/{style}_Before.jpg').shape == edited.shape
    rows = {}
    for name, original in inputs.items():
        truth = load(f'example_data/GT/{name}.jpg')
        assert truth.shape == original.shape, (name, original.shape, truth.shape)
        scaled_truth = finish_spatial(original, truth, STRENGTH)
        masks = target_regions(original, np.zeros(original.shape[:2], bool))
        computed = {}
        for style in styles:
            with Image.open(out / f'{name}-{style}.png') as image:
                computed[style] = np.asarray(image, dtype=np.int16)
        rows[name] = {
            label: {
                'pixels': int(mask.sum()),
                'original_mae': float(np.abs(original[mask].astype(np.int16) - scaled_truth[mask]).mean()),
                'outputs_mae': {style: float(np.abs(image[mask] - scaled_truth[mask]).mean())
                                for style, image in computed.items()},
                'reference_swap_mae': float(np.abs(computed[STYLE[0]][mask] - computed[STYLE[1]][mask]).mean()),
            }
            for label, mask in masks.items() if label in ('whole', 'neutral_proxy') and mask.any()
        }
    protocol = Path(__file__).with_name('inretouch-public-pilot-protocol.md')
    report = dict(tree_sha=TREE, baseline_checkpoint_sha256=engine.checkpoint_sha, strength=STRENGTH,
                  inputs=hashes, predictions=predictions, predictions_sha256=digest(prediction_manifest),
                  cases=rows, model_and_render_seconds=elapsed, protocol_sha256=digest(protocol),
                  script_sha256=digest(__file__))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('tree', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.source, args.tree, args.out)
