"""Range-read the fixed 16-bit MLS pilot from the official FTP archive."""
import argparse
from ftplib import FTP
import hashlib
import io
import json
from pathlib import Path
import struct
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
import zlib

from fetch_mls_light_pilot import HOST


ARCHIVE = '/mls-dataset/images_16bit_color.zip'
SIZE = 2000043792
INDEX_SHA = '8a290dcaaf81d8e0bbacba7a6433e942bf8770137605e98b165cec36f194a5a8'
SCENES = ('02', '03')
LIGHTS = ('2HAL_DESK', '2HAL_DESK_LED-B025')
BUDGET = 32*1024**2


def fetch(first, length):
    with FTP(HOST, timeout=30) as ftp:
        ftp.login()
        ftp.voidcmd('TYPE I')
        assert ftp.size(ARCHIVE) == SIZE
        with ftp.transfercmd(f'RETR {ARCHIVE}', rest=first) as connection:
            with connection.makefile('rb') as source:
                value = source.read(length)
        assert len(value) == length
        ftp.close()
    return value


def run(out, scenes=SCENES):
    out.mkdir(exist_ok=True)
    eocd = fetch(SIZE-22, 22)
    sig, disk, start_disk, disk_count, count, length, offset, comment = struct.unpack('<4s4H2LH', eocd)
    assert sig == b'PK\x05\x06' and disk == start_disk == comment == 0 and disk_count == count
    assert offset + length + 22 == SIZE and length < 512*1024
    index = fetch(offset, length+22)
    assert index[-22:] == eocd
    if hashlib.sha256(index).hexdigest() != INDEX_SHA:
        raise ValueError('Official 16-bit archive index changed')
    with ZipFile(io.BytesIO(index)) as zip_file:
        entries = zip_file.infolist()
    assert len(entries) == count
    members = {Path(entry.filename).name: entry for entry in entries if not entry.is_dir()}
    fixed = [members[f'{scene}_{light}.png'] for scene in scenes for light in LIGHTS]
    total = 22 + len(index) + sum(entry.compress_size+1024 for entry in fixed)
    assert total <= BUDGET
    records = {}
    for entry in fixed:
        position = entry.header_offset+offset
        path = out/Path(entry.filename).name
        received = 0
        if path.exists():
            photo = path.read_bytes()
        else:
            raw = fetch(position, entry.compress_size+1024)
            received = len(raw)
            signature, _, flags, method, _, _, _, _, _, name_len, extra = struct.unpack('<4s5H3L2H', raw[:30])
            start = 30+name_len+extra
            assert signature == b'PK\x03\x04' and not flags & 1 and start <= 1024
            assert method == entry.compress_type and raw[30:30+name_len].decode() == entry.filename
            assert method in (ZIP_DEFLATED, ZIP_STORED)
            payload = raw[start:start+entry.compress_size]
            photo = zlib.decompress(payload, -15) if method == ZIP_DEFLATED else payload
        assert len(photo) == entry.file_size and zlib.crc32(photo) == entry.CRC
        if not path.exists():
            path.write_bytes(photo)
        records[path.name] = dict(sha256=hashlib.sha256(photo).hexdigest(), crc32=entry.CRC,
                                  bytes=len(photo), received_bytes=received)
        print(path.name, received, flush=True)
    report = dict(archive_size=SIZE, index_sha256=hashlib.sha256(index).hexdigest(),
                  max_requested_bytes=total, entries=records)
    (out/'intake.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--wood', action='store_true', help='Read the fixed 13/22 wood scenes instead of 02/03')
    args = parser.parse_args()
    run(args.out, ('13', '22') if args.wood else SCENES)
