"""Read only MMArt-Bench ZIP directory metadata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from urllib.request import ProxyHandler, Request, build_opener
import zipfile

from probe_facd_range import ArchiveWindow


REVISION = '54b33d56f8dabcc2065670a3060063d743f69e65'
URL = f'https://huggingface.co/datasets/JarvisArt/MMArt-Bench/resolve/{REVISION}/MMart-Bench.zip'
TOTAL = 1_439_614_823
TAIL = 65_557
BUDGET = 512 * 1024


def get_range(opener, start, length):
    end = start + length - 1
    headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
    with opener.open(Request(URL, headers=headers), timeout=20) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{TOTAL}':
            raise ValueError('Server did not honor the bounded ZIP range')
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('Incomplete ZIP range')
    return data


def directory(opener):
    tail = get_range(opener, TOTAL - TAIL, TAIL)
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError('Missing end of ZIP directory')
    _, disk, central_disk, per_disk, count, size, offset, comment = struct.unpack_from('<4s4H2IH', tail, eocd)
    if (disk or central_disk or per_disk != count or eocd + 22 + comment != len(tail)
            or offset + size != TOTAL - TAIL + eocd or not 0 < size <= BUDGET - TAIL):
        raise ValueError('Unexpected ZIP directory layout or size')
    central = get_range(opener, offset, size)
    return zipfile.ZipFile(ArchiveWindow(offset, central + tail[eocd:])), count, tail, central


def read_metadata(opener, archive):
    info = archive.getinfo('MMart-Bench/dataInfo.json')
    if info.flag_bits & 1 or not 0 < max(info.file_size, info.compress_size) <= 256 * 1024:
        raise ValueError('Unexpected metadata member')
    header = get_range(opener, info.header_offset, 30)
    if not header.startswith(b'PK\x03\x04'):
        raise ValueError('Missing metadata header')
    name, extra = struct.unpack_from('<2H', header, 26)
    length = 30 + name + extra + info.compress_size
    if length > 300 * 1024 or info.header_offset + length > archive.start_dir:
        raise ValueError('Metadata exceeds range bounds')
    old_file = archive.fp
    data = get_range(opener, info.header_offset, length)
    try:
        archive.fp = ArchiveWindow(info.header_offset, data)
        raw = archive.read(info)
    finally:
        archive.fp = old_file
    return raw, dict(bytes_requested=len(header) + len(data), bytes=len(raw),
                     sha256=hashlib.sha256(raw).hexdigest(), crc32=f'{info.CRC:08x}')


def summarize_pairs(metadata, members):
    rows = json.loads(metadata)
    by_path = {row['path']: row for row in members}
    sources = {row['source_path'] for row in rows}
    targets = {row['gt_path'] for row in rows}
    masks = {PurePosixPath(path).stem for path in by_path if '/local_mask/' in path}
    if (len(sources) != len(rows) or len(targets) != len(rows)
            or not (sources | targets) <= by_path.keys()):
        raise ValueError('Missing or duplicated original/edited file references')
    links = Counter()
    counts = Counter()
    for row in rows:
        source = PurePosixPath(row['source_path'])
        if source.name != PurePosixPath(row['gt_path']).name:
            raise ValueError('Source and edited filenames disagree')
        possible = [mask for mask in masks if source.stem.endswith('_' + mask)]
        has_box = 'box_info' in row
        counts[f'{int(has_box)}_box_{len(possible)}_matches'] += 1
        if has_box:
            links.update(possible)
    signatures = Counter((by_path[path]['size'], by_path[path]['crc32']) for path in sources)
    texts = {' '.join(row['instruction'].casefold().split()) for row in rows}
    return dict(rows=len(rows), source_paths=len(sources), edited_paths=len(targets), mask_files=len(masks),
                fields=dict(Counter(field for row in rows for field in row)),
                box_and_mask_match_cases=dict(counts), matched_mask_files=len(links),
                masks_linked_once=sum(n == 1 for n in links.values()),
                crop_modes=dict(Counter(row['crop_mode'] for row in rows)),
                duplicate_source_size_crc_groups=sum(n > 1 for n in signatures.values()),
                unique_normalized_instructions=len(texts))


def run(out, proxy):
    opener = build_opener(ProxyHandler({'https': proxy})) if proxy else build_opener()
    out.mkdir(exist_ok=False)
    archive, count, tail, central = directory(opener)
    if len(archive.infolist()) != count or len(set(archive.namelist())) != count:
        raise ValueError('Unreliable ZIP member count')
    members = []
    path_counts = Counter()
    extension_counts = Counter()
    for item in archive.infolist():
        if item.is_dir():
            continue
        path = PurePosixPath(item.filename)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe ZIP member name')
        path_counts['/'.join(path.parts[:2])] += 1
        extension_counts[path.suffix.lower()] += 1
        members.append(dict(path=item.filename, size=item.file_size, crc32=f'{item.CRC:08x}'))
    for name, data in [('tail', tail), ('directory', central)]:
        (out / f'{name}.bin').write_bytes(data)
    manifest = out / 'private-members.json'
    manifest.write_text(json.dumps(members, indent=2), encoding='utf-8')
    metadata, identity = read_metadata(opener, archive)
    (out / 'private-dataInfo.json').write_bytes(metadata)
    summary = dict(revision=REVISION, archive_bytes=TOTAL, entries=count, files=len(members),
                   extensions=dict(extension_counts), path_counts=dict(path_counts),
                   pieces={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                           for name, value in [('tail', tail), ('directory', central)]},
                   member_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   metadata=identity, pairs=summarize_pairs(metadata, members), media_read=False,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(entries=count, extensions=summary['extensions'],
                          bytes=len(tail) + len(central) + identity['bytes_requested'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    run(args.out, args.proxy)
