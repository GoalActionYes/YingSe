"""Check measured illuminants and both cameras with only small archive members."""
from argparse import ArgumentParser
from ast import literal_eval
from hashlib import sha256
import json
from math import dist, log2
from pathlib import Path
import re
import struct
import subprocess

from audit_beyond_rgb_lab_index import CENTRAL_SHA, PATTERN, SPLIT_SIZE, zip64_offset
from audit_beyond_rgb_light_pairs import measure
from fetch_beyond_rgb_captures import extract


SCENES = ('WHITE_green', 'WHITE_red')
LIGHTS = (1, 8)
CAMERAS = ('oppo', 'samsung')
PROBE = 'CRI/CRI.txt'
ROLES = (PROBE, *(f'WT/{camera}_tags.json' for camera in CAMERAS))
TAG_BYTES = 4096


def select(path, scenes=SCENES, roles=ROLES):
    data = path.read_bytes()
    if sha256(data).hexdigest() != CENTRAL_SHA:
        raise ValueError('Unexpected central directory hash')
    captures = {}
    pos = 0
    while pos < len(data):
        header = struct.unpack_from('<4s6H3I5H2I', data, pos)
        if header[0] != b'PK\x01\x02':
            raise ValueError('Unexpected central header')
        nlen, xlen, clen = header[10:13]
        first = pos + 46
        name = data[first:first + nlen].decode()
        extra = data[first + nlen:first + nlen + xlen]
        pos = first + nlen + xlen + clen
        match = PATTERN.fullmatch(name)
        if match is None:
            continue
        scene, _, lamp, role = match.groups()
        light = int(lamp)
        if scene not in scenes or light not in LIGHTS or role not in roles:
            continue
        offset = header[-1]
        volume, start = divmod(zip64_offset(extra) if offset == 0xffffffff else offset, SPLIT_SIZE)
        row = captures.setdefault((scene, light), {})
        if role in row:
            raise ValueError('Duplicate requested ZIP member')
        row[role] = dict(name=name, offset=start, volume=volume + 1, size=header[9],
                         compressed_bytes=header[8], crc32=f'{header[7]:08x}')
    if len(captures) != len(scenes) * len(LIGHTS) or any(set(group) != set(roles) for group in captures.values()):
        raise ValueError('Unexpected fixed member selection')
    return captures


def read_tags(folder, stem, member, proxy):
    start = member['offset']
    stop = start + TAG_BYTES - 1
    if stop >= SPLIT_SIZE or member['compressed_bytes'] >= TAG_BYTES - 128:
        raise ValueError('Tags are outside the fixed range budget')
    path = folder / f'{stem}.range'
    headers = folder / f'{stem}-headers.txt'
    fresh = not path.is_file()
    working = path.with_suffix('.partial') if fresh else path
    if fresh:
        url = f"https://zenodo.org/records/16848482/files/beyondRGB.zip.{member['volume']:03d}?download=1"
        subprocess.run(['curl.exe', '--proxy', proxy, '--fail', '--silent', '--show-error',
                        '--max-time', '30', '--range', f'{start}-{stop}', '--max-filesize', str(TAG_BYTES),
                        '--dump-header', str(headers), '--output', str(working), url], check=True)
    response = headers.read_text()
    if (not re.search(r'(?im)^HTTP/\S+\s+206\b', response)
            or not re.search(rf'(?im)^content-range: bytes {start}-{stop}/{SPLIT_SIZE}\s*$', response)):
        raise ValueError('Invalid partial response')
    data = working.read_bytes()
    if len(data) != TAG_BYTES:
        raise ValueError('Wrong tags range length')
    payload = extract(data, 0, member)
    if fresh:
        working.replace(path)
    tags = json.loads(payload)
    exif = tags.get('ExifTag', {})
    if isinstance(exif, str):
        exif = literal_eval(exif)
    settings = {}
    for key in ('ExposureTime', 'ISOSpeedRatings', 'FNumber', 'FocalLength'):
        value = exif.get(key, tags.get(key))
        if value is not None:
            if isinstance(value, str):
                value = literal_eval(value)
            number = value[0] / value[1] if isinstance(value, (tuple, list)) else float(value)
            settings[key] = number
    return dict(model=tags.get('Model'), width=tags.get('ImageWidth'), height=tags.get('ImageLength'),
                settings=settings, sha256=sha256(payload).hexdigest())


def check_camera(records):
    first, second = records
    settings = [item['settings'] for item in records]
    required = {'ExposureTime', 'ISOSpeedRatings', 'FNumber', 'FocalLength'}
    comparable = all(required <= row.keys() and all(row[k] > 0 for k in required) for row in settings)
    capture_matches = all(first[key] is not None and first[key] == second[key] for key in ('model', 'width', 'height'))
    difference = None
    if comparable:
        capture_matches = capture_matches and settings[0]['FocalLength'] == settings[1]['FocalLength']
        levels = [log2(row['ExposureTime'] * row['ISOSpeedRatings'] / row['FNumber'] ** 2) for row in settings]
        difference = abs(levels[0] - levels[1])
    return dict(capture_matches=capture_matches, exposure_difference_ev=difference,
                passes=capture_matches and difference is not None and difference <= .10)


def run(central, cache, out, proxy):
    captures = select(central)
    cache.mkdir(exist_ok=True)
    records = {}
    for scene in SCENES:
        light_readings, tags, members = [], {camera: [] for camera in CAMERAS}, {}
        for light in LIGHTS:
            member = captures[scene, light]
            light_readings.append(measure(cache, scene, light, member[PROBE], proxy))
            for camera in CAMERAS:
                tags[camera].append(read_tags(cache, f'{scene}-{light}-{camera}',
                                              member[f'WT/{camera}_tags.json'], proxy))
            members[light] = member
        lux = [row['lux'] for row in light_readings]
        ratio = max(lux) / min(lux)
        gap = dist(*(row['xy'] for row in light_readings))
        records[scene] = dict(readings=light_readings, lux_ratio=ratio, xy_distance=gap,
                              light_pass=ratio <= 1.15 and .025 <= gap <= .20,
                              cameras={camera: check_camera(rows) | {'tags': rows} for camera, rows in tags.items()},
                              members=members)
    eligible = [camera for camera in CAMERAS if all(row['light_pass'] and row['cameras'][camera]['passes']
                                                    for row in records.values())]
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(dict(central_sha256=CENTRAL_SHA, lights=LIGHTS,
                                     eligible=eligible, scenes=records), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('central', 'cache', 'out'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
