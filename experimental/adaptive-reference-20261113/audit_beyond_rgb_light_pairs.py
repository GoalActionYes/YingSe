"""Select close-illuminance capture pairs using pinned instrument records."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
import json
from math import dist
from pathlib import Path
import re
import struct
import subprocess
import zlib

from audit_beyond_rgb_lab_index import CENTRAL_SHA, MEMBERS, PATTERN, SPLIT_SIZE, WANTED, zip64_offset


def members(data):
    if sha256(data).hexdigest() != CENTRAL_SHA:
        raise ValueError('Changed ZIP directory')
    result = {}
    pos = 0
    while pos < len(data):
        header = struct.unpack_from('<4s6H3I5H2I', data, pos)
        if header[0] != b'PK\x01\x02':
            raise ValueError('Unexpected directory header')
        length, extra, comment = header[10:13]
        start = pos + 46
        name = data[start:start + length].decode()
        end = start + length + extra
        match = PATTERN.fullmatch(name)
        if match:
            scene, capture, lighting, role = match.groups()
            light = int(lighting)
            if scene in WANTED and 1 <= int(capture.split('_')[0]) <= 12 and role in MEMBERS:
                offset = header[-1]
                if offset == 0xffffffff:
                    offset = zip64_offset(data[start + length:end])
                volume, beginning = divmod(offset, SPLIT_SIZE)
                result.setdefault((scene, light), {})[role] = dict(
                    name=name, compressed_bytes=header[8], size=header[9],
                    crc32=f'{header[7]:08x}', volume=volume + 1, offset=beginning)
        pos = end + comment
    if len(result) != 24 or any(set(capture) != set(MEMBERS) for capture in result.values()):
        raise ValueError('Incomplete capture inventory')
    return result


def measure(folder, scene, light, role, proxy):
    key = f'{scene}-light{light:02d}'
    part = folder / f'{key}.range'
    headers = folder / f'{key}-headers.txt'
    start = role['offset']
    stop = start + 2047
    fresh = not part.is_file()
    payload_file = part.with_suffix('.partial') if fresh else part
    if fresh:
        url = f"https://zenodo.org/records/16848482/files/beyondRGB.zip.{role['volume']:03d}?download=1"
        subprocess.run(['curl.exe', '--proxy', proxy, '--fail', '--silent', '--show-error',
                        '--max-time', '25', '--retry', '1', '--retry-max-time', '45',
                        '--range', f'{start}-{stop}', '--max-filesize', '2048', '--dump-header', str(headers),
                        '--output', str(payload_file), url], check=True)
    response = headers.read_text()
    if not re.search(r'(?im)^HTTP/\S+\s+206\b', response):
        raise ValueError('Expected a partial HTTP response')
    if not re.search(rf'(?im)^content-range: bytes {start}-{stop}/{SPLIT_SIZE}\s*$', response):
        raise ValueError('Unexpected range response')
    data = payload_file.read_bytes()
    header = struct.unpack_from('<4s5H3I2H', data)
    name_length, extra_length = header[-2:]
    if len(data) != 2048 or header[0] != b'PK\x03\x04' or header[3] != 8:
        raise ValueError('Invalid ZIP member header')
    if data[30:30 + name_length].decode() != role['name']:
        raise ValueError('Wrong ZIP member name')
    payload = 30 + name_length + extra_length
    raw = zlib.decompress(data[payload:payload + role['compressed_bytes']], -15)
    if len(raw) != role['size'] or f'{zlib.crc32(raw):08x}' != role['crc32']:
        raise ValueError('Wrong ZIP member payload')
    x, y, lux = (float(re.search(rf'(?m)^\s+{label}:\s+([0-9.]+)', raw.decode()).group(1))
                 for label in ('x', 'y', 'Y'))
    if fresh:
        payload_file.replace(part)
    return dict(light=light, xy=[x, y], lux=lux, sha256=sha256(raw).hexdigest())


def compare(readings):
    result = {}
    for a, b in combinations(readings, 2):
        ratio = max(a['lux'], b['lux']) / min(a['lux'], b['lux'])
        gap = dist(a['xy'], b['xy'])
        if ratio <= 1.15 and gap >= .025:
            result[a['light'], b['light']] = dict(lights=[a['light'], b['light']], ratio=ratio, xy_distance=gap)
    return result


def run(central, folder, out, proxy):
    captures = members(central.read_bytes())
    folder.mkdir(exist_ok=True)
    readings = {scene: [measure(folder, scene, light, captures[scene, light]['CRI/CRI.txt'], proxy)
                        for group, light in sorted(captures) if group == scene] for scene in WANTED}
    qualified = {scene: compare(values) for scene, values in readings.items()}
    common = set.intersection(*(set(entries) for entries in qualified.values()))
    choice = {}
    if common:
        best = min(common, key=lambda key: (-min(row[key]['xy_distance'] for row in qualified.values()), key))
        choice = {scene: best for scene in WANTED}
    else:
        for scene, entries in qualified.items():
            if entries:
                choice[scene] = min(entries, key=lambda key: (-entries[key]['xy_distance'], key))
    report = dict(central_sha256=CENTRAL_SHA, readings=readings,
                  qualified={scene: list(values.values()) for scene, values in qualified.items()},
                  selected={scene: {'pair': qualified[scene][pair],
                                    'members': {light: captures[scene, light] for light in pair}}
                            for scene, pair in choice.items()},
                  common_pairs=len(common))
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for arg in ('central', 'folder', 'out'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
