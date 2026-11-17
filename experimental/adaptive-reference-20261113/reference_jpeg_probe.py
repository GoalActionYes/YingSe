"""Measure reference-only JPEG sensitivity on fixed, consumed development pairs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from background_intervention import region_delta, target_regions
from independent_coco import ROOT, digest
from spatial_render import SpatialEngine
from spatial_store import SpatialStore

BASE = ROOT / 'outputs/adaptive-reference-20261113'
PARENT = BASE / 'background-intervention-20261116-v2'
OUT = BASE / 'reference-jpeg-20261118-v1'
SETTINGS = [('jpeg92_444', 92, 0), ('jpeg70_420', 70, 2)]
PAIRS = ['785-3553', '9483-19432', '872-1268', '8021-8690', '285-1993', '2149-2592']


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def main():
    import torch

    torch.set_num_threads(4)
    parent_file = PARENT / 'results.json'
    parent = json.loads(parent_file.read_text(encoding='utf-8'))
    assert parent['finished'] and parent['strength'] == .8
    assert [case['pair'] for case in parent['cases']] == PAIRS
    OUT.mkdir(exist_ok=False)
    report = dict(parent_sha256=digest(parent_file), source_sha256=digest(__file__),
                  pillow=pillow_version, settings=SETTINGS, strength=.8, cases=[])
    (OUT / 'config.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == parent['model_sha256']
    report['model_sha256'] = engine.checkpoint_sha
    for case in parent['cases']:
        pair = case['pair']
        baseline = next(v for v in case['variants'] if v['name'] == 'identity')
        output_file = baseline['methods']['adaptive']['file']
        for item in (case['input_file'], baseline['reference'], output_file):
            assert digest(PARENT / item['path']) == item['sha256']
        target_path = PARENT / case['input_file']['path']
        reference_path = PARENT / baseline['reference']['path']
        target, reference = rgb(target_path), rgb(reference_path)
        # Reconstruct the region mask from its verified original file.
        target_id = pair.split('-')[0]
        mask_name = target_id + '-human-mask.png'
        mask_path = BASE / 'independent-coco' / mask_name
        assert digest(mask_path) == case['parents'][mask_name]
        with Image.open(mask_path) as mask:
            human = np.asarray(mask.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
        regions = target_regions(target, human)
        folder = OUT / pair
        folder.mkdir()
        store = SpatialStore(folder / 'store', engine)
        target_media = store.add_media(target_path)

        def render(path):
            media = store.add_media(path)
            mapping = store.analyze(target_media['id'], media['id'])
            assert not mapping['diagnostics']['certified_aligned']['accepted']
            return store.render(mapping['id'], target, .8), store.decode(media)[0]

        expected = rgb(PARENT / output_file['path'])
        control, decoded = render(reference_path)
        assert np.array_equal(decoded, reference) and np.array_equal(control, expected), pair
        variants = []
        for name, quality, subsampling in SETTINGS:
            jpeg = folder / (name + '.jpg')
            Image.fromarray(reference).save(jpeg, quality=quality, subsampling=subsampling)
            output, actual_reference = render(jpeg)
            assert np.array_equal(actual_reference, rgb(jpeg))
            path = folder / (name + '.png')
            Image.fromarray(output).save(path)
            variants.append(dict(name=name, reference_sha256=digest(jpeg), output_sha256=digest(path),
                                 reference_mae=float(np.abs(reference.astype(float) - actual_reference).mean()),
                                 regions={key: region_delta(output, control, mask) for key, mask in regions.items()}))
        report['cases'].append(dict(pair=pair, variants=variants))
    report['finished'] = True
    (OUT / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Verified {len(PAIRS)} baseline replays and saved {2 * len(PAIRS)} JPEG-reference outputs.")


if __name__ == '__main__':
    main()
