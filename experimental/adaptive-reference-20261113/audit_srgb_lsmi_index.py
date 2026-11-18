"""Summarize saved range responses without reading ZIP members."""

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path
import re
from struct import Struct


TOTAL = 11401528965
ETAG = '"2a7958685-64494a8316c6f"'
CENTRAL = Struct('<4s6H3I5H2I')
ENTRY = re.compile(r'^srgb-lsmi-nikon/(input(?:_masked)?|target(?:_masked)?)/(Place\d+)_(\d+)_(.+)$')


def headers(path):
    response = path.read_text(encoding='ascii').split('HTTP/')[-1].splitlines()
    return int(response[0].split()[1]), {
        key.lower(): value.strip() for key, value in (line.split(':', 1) for line in response[1:] if ':' in line)
    }


def checked_range(root, name, start, end):
    status, fields = headers(root / f'{name}-head.txt')
    data = (root / f'{name}.bin').read_bytes()
    if (status, fields.get('content-range'), fields.get('etag'), len(data)) != (
        206, f'bytes {start}-{end}/{TOTAL}', ETAG, end - start + 1
    ):
        raise ValueError(f'Range response does not match requested archive slice: {name}')
    return data


def central_entries(data):
    offset = data.index(b'PK\x01\x02')
    while offset + CENTRAL.size <= len(data):
        entry = CENTRAL.unpack_from(data, offset)
        if entry[0] != b'PK\x01\x02':
            raise ValueError('Unexpected central directory signature')
        start = offset + CENTRAL.size
        length, extra, comment = entry[10:13]
        offset = start + length + extra + comment
        if offset > len(data):
            break
        yield data[start:start + length].decode('utf-8'), entry[8]


def audit(root):
    files = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    transferred = sum(map(len, files.values()))
    if transferred > 1024 * 1024:
        raise ValueError('Metadata budget exceeded')
    commit = json.loads(files['commit.json'])
    tree = json.loads(files['tree.json'])
    assert tree['sha'] == commit['commit']['tree']['sha'] and not tree['truncated']
    objects = {row['path']: row for row in tree['tree'] if row['type'] == 'blob'}
    for name in ('data.py', 'README.md', 'LICENSE'):
        raw = files[name]
        blob = sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        assert blob == objects[name]['sha']

    tail = checked_range(root, 'nikon-tail', TOTAL - 1024, TOTAL - 1)
    end = Struct('<4sQ2H2I4Q').unpack_from(tail, tail.index(b'PK\x06\x06'))
    assert end[4:6] == (0, 0) and end[6] == end[7]
    count, size, start = end[-3:]
    ranges = {f'nikon-central-{offset}': (start + offset, start + offset + 65535)
              for offset in (1300000, 1450000, 1580000)}
    ranges['nikon-central-start'] = (start, start + 65535)
    ranges['nikon-central-end'] = (start + size - 262144, start + size - 1)
    retained, sizes, entries = {}, {}, {}
    for name, (lo, hi) in ranges.items():
        data = checked_range(root, name, lo, hi)
        records = list(central_entries(data))
        retained[name] = len(records)
        for path, uncompressed_size in records:
            match = ENTRY.fullmatch(path)
            if match:
                role, scene, lighting, _ = match.groups()
                key = (scene, lighting)
                entries.setdefault(role, set()).add(key)
                sizes.setdefault(role, []).append(uncompressed_size)

    archives = {}
    for camera in ('nikon', 'sony'):
        status, fields = headers(root / f'{camera}-head.txt')
        assert status == 200 and fields['accept-ranges'] == 'bytes'
        archives[camera] = dict(length=int(fields['content-length']), etag=fields['etag'])
    assert archives['nikon'] == dict(length=TOTAL, etag=ETAG)
    return dict(commit=commit['sha'], tree=tree['sha'], archives=archives, total_saved_bytes=transferred,
                nikon_zip64=dict(records=count, size=size, offset=start), partial_complete_records=retained,
                roles={role: len(values) for role, values in sizes.items()},
                matched_unmasked=len(entries['input'] & entries['target']),
                matched_masked=len(entries['input_masked'] & entries['target_masked']),
                uncompressed_size_range={role: [min(values), max(values)] for role, values in sizes.items()},
                source_split_sizes={camera: objects[f'metadata/{camera}_split.json']['size']
                                    for camera in ('nikon', 'sony')},
                sha256={name: sha256(data).hexdigest() for name, data in sorted(files.items())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('responses', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.responses), indent=2))
