"""Index four small Beyond RGB capture groups from a pinned central directory."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

CENTRAL_SHA = '203767f1f2b2895e19ce55f330cba1b3b2bfac477227067d7a7708b50efc0e9f'
SPLIT_SIZE = 39_845_888_000
MEMBERS = ('CRI/CRI.txt', 'WT/oppo.h5', 'WT/oppo_cc_detection.json', 'WT/oppo_tags.json')
WANTED = ('MIXED_organic', 'WHITE_rich')
PATTERN = re.compile(r'beyond-unzip/beyondRGB/clb/([^/]+)/([0-9]+_light([0-9]+))/(.+)')


def zip64_offset(extra):
    start = 0
    while start < len(extra):
        kind, size = struct.unpack_from('<HH', extra, start)
        if kind == 1 and size == 8:
            return struct.unpack_from('<Q', extra, start + 4)[0]
        start += 4 + size
    raise ValueError('Expected single ZIP64 offset')


def run(central, out):
    data = central.read_bytes()
    if sha256(data).hexdigest() != CENTRAL_SHA:
        raise ValueError('Unexpected central directory')
    position, count = 0, 0
    lights = defaultdict(set)
    selected = {}
    while position < len(data):
        row = struct.unpack_from('<4s6H3I5H2I', data, position)
        signature, count_before = row[0], count
        if signature != b'PK\x01\x02':
            raise ValueError(f'Invalid header: {count_before}')
        name_size, extra_size, comment_size = row[10:13]
        first = position + 46
        extra_start = first + name_size
        name = data[first:extra_start].decode()
        position = extra_start + extra_size + comment_size
        count += 1
        match = PATTERN.fullmatch(name)
        if not match:
            continue
        scene, capture, light, member = match.groups()
        if member == MEMBERS[0]:
            lights[scene].add(capture)
        if scene in WANTED and light in ('1', '12') and member in MEMBERS:
            offset = row[-1]
            if offset == 0xffffffff:
                offset = zip64_offset(data[extra_start:extra_start + extra_size])
            part, part_offset = divmod(offset, SPLIT_SIZE)
            selected[name] = dict(crc32=f'{row[7]:08x}', compressed_bytes=row[8], size=row[9],
                                  offset=offset, volume=part + 1, volume_offset=part_offset)
    if position != len(data) or count != 30888 or len(selected) != 16:
        raise ValueError('Incomplete directory or selection')
    report = dict(archive_record=16848482, central_sha256=CENTRAL_SHA, member_count=count,
                  capture_counts={scene: len(items) for scene, items in sorted(lights.items())},
                  selected=selected)
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('central', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
