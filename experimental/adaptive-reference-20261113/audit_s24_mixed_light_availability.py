"""Verify byte-bounded S24 secondary-light metadata and mask evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fetch_s24_calibrated_pilot import REVISION, sha


BLANK = '78fce7ca67e4c1f1c50e70e6d44ac35253c0e89e06ac8578cda6e2cb5b22429b'


def verify(folder):
    roles = {}
    for role, ext in (('masks', 'png'), ('data', 'json')):
        rows = json.loads((folder / f'{role}.json').read_bytes())
        assert len(rows) == 100 and {r['path'] for r in rows} == {
            f'train/{role}/img_{i:04d}.{ext}' for i in range(100)}
        roles[role] = {Path(row['path']).stem: row for row in rows}
    retained = sorted(key for key, row in roles['masks'].items() if row.get('lfs', {}).get('oid') != BLANK)[:6]
    plan = json.loads((folder / 'plan.json').read_bytes())
    assert plan == [roles[role][key] for role in ('masks', 'data') for key in retained]
    rows = {}
    for key in retained:
        records = {}
        for role, ext in (('masks', 'png'), ('data', 'json')):
            entry = roles[role][key]
            data = (folder / f'{key}.{ext}').read_bytes()
            assert len(data) == entry['size'] and len(data) <= 512*1024
            if 'lfs' in entry:
                assert sha(data) == entry['lfs']['oid']
            else:
                assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == entry['oid']
            records[role] = dict(bytes=len(data), sha256=sha(data))
        with Image.open(folder / f'{key}.png') as image:
            mask = np.asarray(image.convert('L'))
        nonzero = mask != 0
        adjacent = int((nonzero[1:] & nonzero[:-1]).sum() + (nonzero[:, 1:] & nonzero[:, :-1]).sum())
        metadata = json.loads((folder / f'{key}.json').read_bytes())
        fields = sorted(metadata)
        shape_ok = sorted(mask.shape) == [1500, 2000]
        coverage = float(nonzero.mean())
        rows[key] = dict(files=records, shape=list(mask.shape), values=np.unique(mask).tolist(),
                         nonzero_pixels=int(nonzero.sum()), adjacent_pairs=adjacent, coverage=coverage,
                         size_eligible=bool(shape_ok and .02 <= coverage <= .8 and adjacent > 0),
                         metadata_fields=fields, capture_fields=sorted(metadata['capture_metadata']),
                         additional_fields=sorted(metadata['additional_metadata']))
    indexes = {role: (folder / f'{role}.json').read_bytes() for role in roles}
    total = sum(len(one) for one in indexes.values()) + sum(
        item['bytes'] for row in rows.values() for item in row['files'].values())
    assert total <= 4*1024**2
    return dict(revision=REVISION, index_sha256={role: sha(value) for role, value in indexes.items()},
                response_bytes=total, plan_sha256=sha((folder / 'plan.json').read_bytes()), cases=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(verify(args.folder), stream, indent=2, allow_nan=False)
        stream.write('\n')
