"""Read and verify a capped set of public FACD members without downloading the archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageOps

from independent_coco import digest
from probe_tst_reference_spatial_signal import structure_score
from reference_color_app.media import decode_image


URL = 'https://drive.usercontent.google.com/download?id=0Bzz3iFiaAkd3SFJXQ3hIY0dQdDg&export=download&confirm=t'
TOTAL = 2_968_999_768
TAIL = 65_557
FILTERS = ('Origin', 'Amaro', 'Hudson', 'XProII')
MIB = 1024 ** 2


def request_range(start, length, proxy):
    end = start + length - 1
    with tempfile.TemporaryDirectory() as temp:
        headers = Path(temp) / 'headers.txt'
        args = ['curl.exe', '--fail', '--silent', '--show-error', '--location',
                '--max-time', '35', '--max-filesize', str(length),
                '--range', f'{start}-{end}', '--dump-header', str(headers)]
        if proxy:
            args += ['--proxy', proxy]
        result = subprocess.run([*args, URL], capture_output=True, check=True)
        header = headers.read_text(encoding='iso-8859-1').strip().split('\n\n')[-1]
        fields = dict((name.lower(), value.strip()) for name, value in re.findall(r'^([^:\n]+): (.*)$', header, re.M))
        if (header.splitlines()[0].split()[1] != '206' or
                fields.get('content-range') != f'bytes {start}-{end}/{TOTAL}' or len(result.stdout) != length):
            raise ValueError(f'Server returned an unexpected byte range for {start}-{end}')
        return result.stdout


class ArchiveWindow:
    """Expose one contiguous slice at its original archive offsets."""
    def __init__(self, offset, data):
        self.offset = offset
        self.data = data
        self.position = offset

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.offset + len(self.data))[whence]
        self.position = base + offset
        if self.position < 0:
            raise OSError('Invalid ZIP archive offset')
        return self.position

    def read(self, size=-1):
        start = self.position - self.offset
        end = len(self.data) if size < 0 else min(len(self.data), start + size)
        if not 0 <= start <= end:
            raise OSError('ZIP read outside selected range')
        self.position += end - start
        return self.data[start:end]

    def seekable(self):
        return True


def get_index(proxy):
    tail = request_range(TOTAL - TAIL, TAIL, proxy)
    fields = struct.unpack_from('<4s4H2IH', tail, len(tail) - 22)
    signature, disk, central_disk, disk_count, count, size, offset, comment = fields
    if (signature != b'PK\x05\x06' or disk or central_disk or disk_count != count or comment
            or offset + size != TOTAL - 22 or size + TAIL > 12 * MIB):
        raise ValueError('Unexpected ZIP index type or size')
    catalog = request_range(offset, size, proxy)
    archive = zipfile.ZipFile(ArchiveWindow(offset, catalog + tail[-22:]))
    if len(archive.infolist()) != count:
        raise ValueError('ZIP directory record count mismatch')
    return archive, {name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                     for name, data in [('tail', tail), ('central', catalog)]}


def dhash(rgb):
    gray = ImageOps.grayscale(Image.fromarray(rgb))
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    vertical = np.asarray(gray.resize((8, 9), Image.Resampling.BILINEAR))
    return np.r_[np.diff(horizontal.astype(np.int16), axis=1).ravel() > 0,
                 np.diff(vertical.astype(np.int16), axis=0).ravel() > 0]


def run(out, proxy):
    out.mkdir(exist_ok=False)
    archive, index = get_index(proxy)
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous directory names')
    members = {kind: {} for kind in FILTERS}
    for item in archive.infolist():
        match = re.fullmatch(r'FACD_image/([^/]+)/(\d+)\.jpg', item.filename)
        if match and match[1] in members:
            members[match[1]][match[2]] = item
    shared = set.intersection(*(set(values) for values in members.values()))
    selected = sorted(shared, key=lambda name: hashlib.sha256(name.encode('utf-8')).hexdigest())[:8]
    if len(selected) != 8:
        raise ValueError('Insufficient shared original-photo IDs')
    originals, cases, jobs = {}, {}, []
    for original_id in selected:
        group = out / original_id
        group.mkdir()
        cases[original_id] = {}
        for kind in FILTERS:
            info = members[kind][original_id]
            if not 0 < max(info.compress_size, info.file_size) <= 2 * MIB:
                raise ValueError(f'Selected member exceeds the limit: {info.filename}')
            span = info.compress_size + 4096
            jobs.append((original_id, kind, info, span))
    if sum(job[3] for job in jobs) > 16 * MIB:
        raise ValueError('Selected ranges exceed the media budget')
    selection = {original_id: [members[kind][original_id].filename for kind in FILTERS]
                 for original_id in selected}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

    worker = lambda job: request_range(job[2].header_offset, job[3], proxy)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for (original_id, kind, info, _), data in zip(jobs, pool.map(worker, jobs), strict=True):
            archive.fp = ArchiveWindow(info.header_offset, data)
            plain = archive.read(info)
            if len(plain) != info.file_size:
                raise ValueError(f'Selected member size mismatch: {info.filename}')
            path = out / original_id / f'{kind}.jpg'
            path.write_bytes(plain)
            rgb = decode_image(path)
            row = dict(sha256=digest(path), bytes=len(plain), crc32=f'{info.CRC:08x}',
                       shape=list(rgb.shape[:2]), member=info.filename)
            if kind == 'Origin':
                originals[original_id] = rgb
            else:
                before = originals[original_id]
                small = [np.asarray(Image.fromarray(value).resize((256, 256), Image.Resampling.BILINEAR))
                         for value in (before, rgb)]
                row['geometry'] = structure_score(*small) if rgb.shape == before.shape else {'passed': False}
            cases[original_id][kind] = row
    hashes = {name: (dhash(rgb), dhash(rgb[:, ::-1])) for name, rgb in originals.items()}
    min_distance = min(int(np.count_nonzero(a != b)) for l, r in combinations(selected, 2)
                       for a in hashes[l] for b in hashes[r])
    exact_distinct = len({cases[name]['Origin']['sha256'] for name in selected}) == len(selected)
    usable = sum(all(row[kind]['geometry']['passed'] for kind in FILTERS[1:]) for row in cases.values())
    archive.fp = None
    result = dict(archive_bytes=TOTAL, index=index, index_members=len(names), shared_ids=len(shared),
                  filter_counts={key: len(values) for key, values in members.items()},
                  selected=selected, cases=cases, minimum_original_dhash_distance=min_distance,
                  exact_originals_distinct=exact_distinct, usable_families=usable,
                  eligible=usable >= 6 and min_distance > 10 and exact_distinct,
                  media_range_bytes=sum(job[3] for job in jobs), script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('facd-range-pilot-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    run(**vars(parser.parse_args()))
