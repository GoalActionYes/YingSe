"""Fetch only the four Oppo captures selected by the measured-light audit."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess
import zlib

from audit_beyond_rgb_lab_index import SPLIT_SIZE


SELECTION_SHA = '1225b2d32061758f0de18a050dec2e061ce172501accab0d5de2123d1bfbc741'
ROLES = ('WT/oppo.h5', 'WT/oppo_cc_detection.json', 'WT/oppo_tags.json')


def extract(data, offset, member):
    header = struct.unpack_from('<4s5H3I2H', data, offset)
    if header[0] != b'PK\x03\x04' or header[3] != 8:
        raise ValueError('Unexpected ZIP member header')
    name_length, extra_length = header[-2:]
    name_start = offset + 30
    if data[name_start:name_start + name_length].decode() != member['name']:
        raise ValueError('Unexpected ZIP member name')
    start = name_start + name_length + extra_length
    raw = zlib.decompress(data[start:start + member['compressed_bytes']], -15)
    if len(raw) != member['size'] or f'{zlib.crc32(raw):08x}' != member['crc32']:
        raise ValueError('Unexpected ZIP member payload')
    return raw


def run(selection, destination, proxy):
    raw = selection.read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != SELECTION_SHA:
        raise ValueError('Selection differs from the frozen light-pair audit')
    captures = {}
    for scene, group in json.loads(raw)['selected'].items():
        for light, catalog in group['members'].items():
            links = [catalog[role] for role in ROLES]
            image, _, tags = links
            start = image['offset']
            # Include the final local header and a bounded allowance for its extra fields.
            stop = tags['offset'] + tags['compressed_bytes'] + 1023
            if any(link['volume'] != image['volume'] for link in links):
                raise ValueError('Capture members span multiple volumes')
            if not all(start <= link['offset'] < stop < SPLIT_SIZE for link in links):
                raise ValueError('Capture range is outside its volume')
            captures[f'{scene}-light{light}'] = (links, start, stop, image['volume'])
    if len(captures) != 4 or sum(stop - start + 1 for _, start, stop, _ in captures.values()) > 72_000_000:
        raise ValueError('Outside the frozen capture or transfer budget')

    for key, (links, start, stop, volume) in captures.items():
        folder = destination / key
        folder.mkdir(parents=True, exist_ok=True)
        report = folder / 'members.json'
        if report.is_file():
            cached = json.loads(report.read_text(encoding='utf-8'))
            for link in links:
                name = Path(link['name']).name
                if sha256((folder / name).read_bytes()).hexdigest() != cached[name]:
                    raise ValueError(f'Cached member hash mismatch in {key}')
            continue
        transfer = folder / 'archive.range'
        headers = folder / 'headers.txt'
        size = stop - start + 1
        url = f'https://zenodo.org/records/16848482/files/beyondRGB.zip.{volume:03d}?download=1'
        subprocess.run(['curl.exe', '--proxy', proxy, '--fail', '--silent', '--show-error',
                        '--max-time', '100', '--range', f'{start}-{stop}', '--max-filesize', str(size),
                        '--dump-header', str(headers), '--output', str(transfer), url], check=True)
        response = headers.read_text()
        if (not re.search(r'(?im)^HTTP/\S+\s+206\b', response)
                or not re.search(rf'(?im)^content-range: bytes {start}-{stop}/{SPLIT_SIZE}\s*$', response)):
            raise ValueError(f'Unexpected range response for {key}')
        data = transfer.read_bytes()
        if len(data) != size:
            raise ValueError(f'Unexpected range length for {key}')
        hashes = {}
        for link in links:
            member = extract(data, link['offset'] - start, link)
            name = Path(link['name']).name
            (folder / name).write_bytes(member)
            hashes[name] = sha256(member).hexdigest()
        temporary = report.with_suffix('.tmp')
        temporary.write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
        temporary.replace(report)
        print(f'{key}: {size:,} bytes verified', flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
