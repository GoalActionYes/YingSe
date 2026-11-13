"""A new stratified original-image split; all augmentations inherit that split."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from independent_coco import ROOT, digest, fetch, select_images, person_annotations, ANNOTATION_SHA, BUCKET


def split_groups(groups, seed=20261113):
    if set(groups) != {'single', 'multiple', 'none'} or any(len(v) != 100 for v in groups.values()):
        raise ValueError('Expected 100 originals per group')
    ids = [x['id'] for v in groups.values() for x in v]
    if len(set(ids)) != len(ids):
        raise ValueError('Original identities overlap')
    rng = random.Random(seed)
    result = {'train': [], 'validation': [], 'test': []}
    for group in ('single', 'multiple', 'none'):
        records = [dict(x, group=group) for x in groups[group]]
        rng.shuffle(records)
        result['train'].extend(records[:80])
        result['validation'].extend(records[80:90])
        result['test'].extend(records[90:])
    return result


def main():
    cache = ROOT / '.downloads/coco'
    archive = cache / 'annotations_trainval2017.zip'
    if digest(archive) != ANNOTATION_SHA:
        raise ValueError('Annotation archive changed')
    old_path = ROOT / 'outputs/adaptive-reference-20261113/independent-coco/manifest.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    excluded_ids = {x['id'] for x in old['images']}
    excluded_sha = {x['original_sha256'] for x in old['images']}
    with zipfile.ZipFile(archive) as z:
        with z.open('annotations/instances_val2017.json') as stream:
            annotations = json.load(stream)
    groups = select_images([x for x in annotations['images'] if x['id'] not in excluded_ids], annotations['annotations'], count=100)
    splits = split_groups(groups)
    people = person_annotations(annotations['annotations'])
    licenses = {x['id']: x for x in annotations['licenses']}
    out = ROOT / 'outputs/adaptive-reference-20261113/supervised-style'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(exist_ok=True)
    def prepare(item):
        split, image = item
        name = image['file_name']
        if Path(name).name != name:
            raise ValueError('Unexpected image path')
        source = cache / 'images' / name
        fetch(BUCKET+'/val2017/'+name, source)
        source_sha = digest(source)
        if source_sha in excluded_sha:
            raise ValueError('Exact prior image content leaked through a different ID')
        rgb = Image.open(source).convert('RGB')
        if rgb.size != (image['width'], image['height']):
            raise ValueError('Annotation/image geometry mismatch')
        mask = Image.new('L', rgb.size)
        draw = ImageDraw.Draw(mask)
        for annotation in people[image['id']]:
            for polygon in annotation['segmentation']:
                if len(polygon) < 6 or len(polygon)%2:
                    raise ValueError('Invalid human polygon')
                draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
        rgb.thumbnail((640, 640), Image.Resampling.BILINEAR)
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
        image_path = out / 'images' / f'{image["id"]}.png'
        mask_path = out / 'masks' / f'{image["id"]}.png'
        rgb.save(image_path)
        mask.save(mask_path)
        return {'id': image['id'], 'split': split, 'group': image['group'],
                'source_url': BUCKET+'/val2017/'+name, 'flickr_url': image.get('flickr_url'),
                'license': licenses[image['license']], 'original_sha256': source_sha,
                'image_file': str(image_path.relative_to(out)), 'mask_file': str(mask_path.relative_to(out)),
                'image_sha256': digest(image_path), 'mask_sha256': digest(mask_path)}
    tasks = [(split, image) for split, images in splits.items() for image in images]
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(prepare, tasks), 1):
            records.append(record)
            if i%20 == 0:
                print(json.dumps({'prepared': i, 'total': 300}), flush=True)
    for key in ('original_sha256', 'image_sha256'):
        if len({r[key] for r in records}) != len(records):
            raise ValueError('Exact duplicate original or decoded content; do not silently reuse across splits')
    report = {'scope': 'Supervised style-identifiability probe, not natural style-transfer acceptance',
              'seed': 20261113, 'excluded_prior_ids': sorted(excluded_ids), 'excluded_prior_manifest_sha256': digest(old_path),
              'annotation_sha256': ANNOTATION_SHA, 'selector_source_sha256': digest(Path(__file__)),
              'protocol_sha256': digest(Path(__file__).with_name('supervised-style-probe.md')),
              'counts': {split: len(items) for split, items in splits.items()}, 'images': records,
              'test_status': 'Not generated or evaluated'}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'exact_content_unique': True}), flush=True)


if __name__ == '__main__':
    main()
