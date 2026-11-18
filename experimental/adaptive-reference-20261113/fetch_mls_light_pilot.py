"""Fetch eight specific MLS JPEG members under a fixed FTP byte budget."""
import argparse
from ftplib import FTP
import hashlib
import io
import json
from pathlib import Path
import struct
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
import zlib


SCENES = ('02', '03', '13', '22')
LIGHTS = ('2HAL_DESK', '2HAL_DESK_LED-B100')
ARCHIVE_SIZE = 93125444
INDEX_SHA = '8845839b4aa6c58822ad01fc842501f38111fa46bec169816e4b88508fcc2b40'
HOST = 'vis.iitp.ru'
ARCHIVE = '/mls-dataset/images_preview.zip'


def fetch(offset, length):
    with FTP(HOST, timeout=25) as ftp:
        ftp.login()
        ftp.voidcmd('TYPE I')
        assert ftp.size(ARCHIVE) == ARCHIVE_SIZE
        with ftp.transfercmd(f'RETR {ARCHIVE}', rest=offset) as connection:
            with connection.makefile('rb') as source:
                value = source.read(length)
        assert len(value) == length
        # The data connection closes at the end of this member.
        ftp.close()
    return value


def run(index_path, out):
    index = index_path.read_bytes()
    assert hashlib.sha256(index).hexdigest() == INDEX_SHA
    with ZipFile(io.BytesIO(index)) as archive:
        members = {entry.filename: entry for entry in archive.infolist()}
    fixed = [members[f'images_preview/{scene}/{scene}_{light}.jpg']
             for scene in SCENES for light in LIGHTS]
    assert sum(entry.compress_size+1024 for entry in fixed) < 4*1024**2
    out.mkdir(exist_ok=True)
    records = {}
    for entry in fixed:
        offset = entry.header_offset + ARCHIVE_SIZE-len(index)
        assert 0 <= offset < ARCHIVE_SIZE-len(index)
        raw = fetch(offset, entry.compress_size+1024)
        sig, _, flags, method, _, _, _, _, _, name_len, extra_size = struct.unpack('<4s5H3L2H', raw[:30])
        start = 30+name_len+extra_size
        assert sig == b'PK\x03\x04' and start <= 1024 and not flags & 1
        assert raw[30:30+name_len].decode() == entry.filename and method == entry.compress_type
        packed = raw[start:start+entry.compress_size]
        assert method in (ZIP_DEFLATED, ZIP_STORED)
        photo = zlib.decompress(packed, -15) if method == ZIP_DEFLATED else packed
        assert len(photo) == entry.file_size and zlib.crc32(photo) == entry.CRC
        name = Path(entry.filename).name
        path = out/name
        if path.exists():
            assert photo == path.read_bytes()
        else:
            path.write_bytes(photo)
        records[name] = dict(sha256=hashlib.sha256(photo).hexdigest(), crc32=entry.CRC,
                             bytes=len(photo), received_bytes=len(raw))
        print(name, flush=True)
    (out/'intake.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.index, args.out)
