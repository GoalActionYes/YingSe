"""Verify two instruction-similar grades and compare bidirectional two-image outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_reference_chroma_spread import projection
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial

IDS = ('102', '105')


def verify_source(folder, task):
    tree = json.loads((folder / f'{task}-tree.json').read_text(encoding='utf-8'))
    roles = ('start.jpg', 'end.jpg', 'rubric.json')
    rows = {Path(entry['path']).name: entry for entry in tree}
    result = {}
    for role in roles:
        name = f'{task}_{role}'
        path, entry = folder / name, rows[name]
        data = path.read_bytes()
        assert len(data) == entry['size']
        sha = digest(path)
        if entry.get('lfs'):
            assert sha == entry['lfs']['oid']
        else:
            assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['oid']
        result[role] = dict(sha256=sha, size=len(data))
    return result


def run(folder, out):
    files = {task: verify_source(folder, task) for task in IDS}
    all_images = {}
    geometry = {}
    scores = {}
    out.mkdir(exist_ok=False)
    for task in IDS:
        native = [decode_image(folder / f'{task}_{role}.jpg') for role in ('start', 'end')]
        small = tuple(np.array(Image.fromarray(value).resize((256, 256), Image.Resampling.BILINEAR)) for value in native)
        geometry[task] = structure_score(*small)
        geometry[task]['native_shapes'] = [list(image.shape[:2]) for image in native]
        all_images[task] = small
    report = dict(files=files, geometry=geometry, predictions={}, scores=scores, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('psbench-shared-local-intent-protocol.md')))
    distinct = len({digest(folder / f'{task}_start.jpg') for task in IDS}) == 2
    distinct = distinct and not np.array_equal(all_images[IDS[0]][0], all_images[IDS[1]][0])
    report['distinct_input_bytes_and_pixels'] = distinct
    if distinct and all(g['passed'] for g in geometry.values()):
        engine = SpatialEngine('multiref-v1')
        report['checkpoint_sha256'] = engine.checkpoint_sha
        for target, reference in (IDS, IDS[::-1]):
            original, _ = all_images[target]
            _, look = all_images[reference]
            rendered = engine.prepare(original, look).apply(original)
            path = out / f'{target}.png'
            Image.fromarray(finish_spatial(original, rendered, .8)).save(path)
            report['predictions'][target] = digest(path)
        (out / 'predictions.json').write_text(json.dumps(report['predictions'], indent=2), encoding='utf-8')
        for target in IDS:
            original, truth = all_images[target]
            expected = finish_spatial(original, truth, .8).astype(np.int16)
            with Image.open(out / f'{target}.png') as image:
                output = np.asarray(image, dtype=np.int16)
            delta = expected - original
            regions = target_regions(original, np.zeros(original.shape[:2], dtype=bool))
            scores[target] = {
                name: dict(original_mae=float(np.abs(delta[mask]).mean()),
                           result_mae=float(np.abs((output - expected)[mask]).mean()),
                           response_projection=projection(output - original, delta, mask))
                for name, mask in regions.items() if name in ('whole', 'neutral_proxy') and mask.any()
            }
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(geometry=geometry, scores=scores)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.folder, args.out)
