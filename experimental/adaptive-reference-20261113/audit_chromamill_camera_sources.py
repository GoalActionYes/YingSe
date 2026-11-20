"""Freeze CHROMA-MILL camera and content groups from published file metadata."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urljoin, urlsplit

from probe_chromamill_capture_cross import REVISION


CAMERAS = ('nikon', 'phone')
PARTITIONS = ('train', 'val', 'test')
TRAIN = ('Scene3_a', 'Scene9_b', 'Scene10_a', 'Scene23_a')
FILE = re.compile(r'((Scene|Test|Val)([0-9]+)(?:_[a-z])?)-F_90-SS_200-x_([0-9]+)-y_([0-9]+)-B_([0-9]+)\.png')
HOST = 'https://huggingface.co'
ROOT = '/api/datasets/priubrogent/CHROMA-MILL/tree'
MEMBER_LIMIT = 700_000


def fetch_index(camera, partition, cache, proxy):
    path = f'{ROOT}/{REVISION}/{camera}/{partition}/low'
    url = f'{HOST}{path}?limit=1000'
    records, pages = {}, []
    for page in range(4):
        location = urlsplit(url)
        if location.scheme != 'https' or location.netloc != 'huggingface.co' or unquote(location.path) != path:
            raise ValueError('Unexpected pagination destination')
        body = cache / f'{camera}-{partition}-{page}.json'
        headers = body.with_suffix('.headers')
        if not body.is_file() or not headers.is_file():
            temporary = body.with_suffix('.partial')
            subprocess.run(['curl.exe', '--proxy', proxy, '--silent', '--fail', '--show-error',
                            '--max-time', '25', '--max-filesize', str(MEMBER_LIMIT),
                            '--dump-header', str(headers), '--output', str(temporary), url], check=True)
            temporary.replace(body)
        raw = body.read_bytes()
        if len(raw) > MEMBER_LIMIT or not isinstance(json.loads(raw), list):
            raise ValueError('Unexpected index response')
        pages.append(dict(sha256=sha256(raw).hexdigest(), bytes=len(raw)))
        for row in json.loads(raw):
            name = Path(row['path']).name
            parsed = FILE.fullmatch(name)
            if parsed is None:
                raise ValueError(f'Unexpected capture naming: {name}')
            scene, prefix, index, x, y, strength = parsed.groups()
            key = scene, f'{x}-{y}', int(strength)
            digest = row['lfs']['oid']
            if key in records or row['size'] != row['lfs']['size'] or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Unexpected LFS file metadata')
            records[key] = dict(path=row['path'], bytes=row['size'], sha256=digest, family=f'{prefix}{index}')
        next_page = re.search(r'(?im)^link:.*?<([^>]+)>;\s*rel="next"', headers.read_text())
        if next_page is None:
            return records, pages
        url = urljoin(url, next_page[1])
    raise ValueError('Metadata page budget exhausted')


def verify_plan(indexes, role, scenes, lights):
    records, missing = {}, []
    for camera, scene, xy in product(CAMERAS, scenes, lights):
        key = scene, xy, 241
        name = f'{camera}/{scene}/{xy}'
        if key in indexes[camera, role]:
            records[name] = indexes[camera, role][key]
        else:
            missing.append(name)
    return records, missing


def run(cache, out, proxy):
    cache.mkdir(exist_ok=True)
    indexes, receipts = {}, {}
    for camera, partition in product(CAMERAS, PARTITIONS):
        rows, pages = fetch_index(camera, partition, cache, proxy)
        key = f'{camera}/{partition}'
        indexes[camera, partition] = rows
        receipts[key] = dict(files=len(rows),
                             scenes=sorted(set(group[0] for group in rows)),
                             families=sorted(set(record['family'] for record in rows.values())),
                             lights=sorted(set(group[1] for group in rows)),
                             strength_counts=dict(sorted(Counter(group[2] for group in rows).items())),
                             pages=pages)
    cross = {}
    for part in PARTITIONS:
        nikon, phone = (indexes[camera, part] for camera in CAMERAS)
        same = nikon.keys() & phone.keys()
        cross[part] = dict(shared_keys=len(same), nikon_only=len(nikon.keys() - phone.keys()),
                           phone_only=len(phone.keys() - nikon.keys()),
                           identical_corresponding_sha=sum(nikon[key]['sha256']==phone[key]['sha256'] for key in same))

    families = {part: set().union(*(set(receipts[f'{camera}/{part}']['families']) for camera in CAMERAS))
                for part in PARTITIONS}
    digests = {part: set().union(*({row['sha256'] for row in indexes[camera, part].values()} for camera in CAMERAS))
               for part in PARTITIONS}
    overlaps = {f'{a}/{b}': dict(families=sorted(families[a] & families[b]),
                                identical_files=len(digests[a] & digests[b]))
                for a,b in (('train','val'),('train','test'),('val','test'))}

    fixed_lights = tuple(receipts['phone/train']['lights'])
    if len(fixed_lights)!=4 or not {'3127-3290','1968-1623'} <= set(fixed_lights):
        raise ValueError('Expected the four calibrated lights')
    names = sorted((name for name in receipts['phone/val']['scenes']
                    if all((name, xy, 241) in indexes[camera,'val'] for camera, xy in product(CAMERAS,fixed_lights))),
                   key=lambda name: (int(re.search(r'\d+', name)[0]), name))
    validation = tuple(list(dict.fromkeys(re.sub(r'_[a-z]$', '', name) for name in names))[:4])
    if len(validation)!=4:
        raise ValueError('Not enough complete validation families')
    train, missing_train = verify_plan(indexes, 'train', TRAIN, fixed_lights)
    val, missing_val = verify_plan(indexes, 'val', validation, fixed_lights)
    total = sum(row['bytes'] for row in train.values())
    new_train = {k: row for k, row in train.items() if not (k.startswith('phone/') and
                 k.rsplit('/',1)[-1] in ('3127-3290','1968-1623'))}
    result = dict(revision=REVISION, receipts=receipts, cross_camera=cross, overlaps=overlaps,
                  selected_train=train, missing_train=missing_train, selected_validation=val,
                  missing_validation=missing_val, cached_train_members=len(train)-len(new_train),
                  train_bytes=total, additional_train_bytes=sum(row['bytes'] for row in new_train.values()),
                  validation_bytes=sum(row['bytes'] for row in val.values()),
                  training_budget=192_000_000)
    with out.open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--proxy',default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
