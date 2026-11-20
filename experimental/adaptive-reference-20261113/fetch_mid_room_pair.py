"""Fetch four fixed MIT Multi-Illumination members by verified HTTP range."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

from probe_facd_range import ArchiveWindow


SCENES = {
    '14n_copyroom1': (11171749, '"aa77a5-5951bee57bdb6"'),
    'everett_living2': (12216135, '"ba6747-59520c3958f5c"'),
}
DIRECTIONS = (14, 24)
TAIL = 65557


def read_range(url, total, etag, start, length):
    end = start + length - 1
    with TemporaryDirectory() as temp:
        header = Path(temp) / 'headers.txt'
        data = subprocess.run(['curl.exe', '-fsS', '--max-time', '55', '--max-filesize', str(length),
                               '--range', f'{start}-{end}', '--dump-header', str(header),
                               '--header', f'If-Range: {etag}', url], capture_output=True, check=True).stdout
        raw = header.read_text('latin-1').strip().split('\n\n')[-1]
        fields = {name.lower(): value.strip() for name, value in re.findall(r'^([^:\n]+): (.*)$', raw, re.M)}
        if (raw.splitlines()[0].split()[1] != '206' or
                fields.get('content-range') != f'bytes {start}-{end}/{total}' or
                fields.get('etag') != etag or len(data) != length):
            raise ValueError('Unexpected public archive range response')
    return data


def extract(scene, out):
    total, etag = SCENES[scene]
    url = f'https://data.csail.mit.edu/multilum/{scene}/{scene}_mip2_jpg.zip'
    start = total - TAIL
    tail = read_range(url, total, etag, start, TAIL)
    archive = ZipFile(ArchiveWindow(start, tail))
    rows = {}
    for direction in DIRECTIONS:
        name = f'{scene}/dir_{direction}_mip2.jpg'
        member = archive.getinfo(name)
        span = member.compress_size + 4096
        if not 0 < member.file_size <= 2 * 1024**2 or member.header_offset + span >= total:
            raise ValueError('Member exceeds download bounds')
        archive.fp = ArchiveWindow(member.header_offset, read_range(url, total, etag, member.header_offset, span))
        data = archive.read(member)
        target = out / f'{scene}-{direction}.jpg'
        target.write_bytes(data)
        with Image.open(target) as image:
            image.load()
            if image.size != (1500, 1000) or image.mode != 'RGB':
                raise ValueError('Unexpected photograph encoding')
        rows[str(direction)] = dict(member=name, bytes=len(data), range_bytes=span, crc=f'{member.CRC:08x}',
                                    sha256=sha256(data).hexdigest())
    archive.fp = None
    return dict(etag=etag, archive_bytes=total, index_sha=sha256(tail).hexdigest(), members=rows)


def run(out):
    out.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        data = dict(zip(SCENES, pool.map(lambda name: extract(name, out), SCENES), strict=True))
    (out / 'results.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
