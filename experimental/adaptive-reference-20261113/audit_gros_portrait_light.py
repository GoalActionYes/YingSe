"""Read the ZIP directory and explicitly selected public portrait JPEGs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
from itertools import repeat
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import zipfile

from PIL import Image


URL = ('https://data.mendeley.com/public-files/datasets/bpy24trj6s/files/'
       '3ff1c176-c7a0-49db-afba-32e368012e83/file_downloaded')
ARCHIVE_SIZE = 7_701_332_033
ARCHIVE_TAG = '"c05296a92d0d90d51d1c26002fb0c957-1469"'
MIB = 1024**2
SEGMENT = 512 * 1024


class PublicRange(io.RawIOBase):
    def __init__(self, limit, proxy):
        super().__init__()
        self.position = 0
        self.transferred = 0
        self.limit = limit
        self.proxy = proxy

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, ARCHIVE_SIZE)[whence] + offset
        if not 0 <= self.position <= ARCHIVE_SIZE:
            raise ValueError('ZIP offset outside archive')
        return self.position

    def read(self, size=-1):
        rest = ARCHIVE_SIZE - self.position
        size = rest if size < 0 else min(size, rest)
        if not size:
            return b''
        if self.transferred + size > self.limit:
            raise ValueError('Archive transfer budget exceeded')
        starts = range(self.position, self.position + size, SEGMENT)
        end = self.position + size
        if len(starts) == 1:
            data = self.read_segment(self.position, end)
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                data = b''.join(pool.map(self.read_segment, starts, repeat(end)))
        self.transferred += size
        self.position += size
        return data

    def read_segment(self, start, stop):
        end = min(stop, start + SEGMENT) - 1
        size = end - start + 1
        with TemporaryDirectory() as scratch:
            headers = Path(scratch) / 'headers.txt'
            arguments = ['curl.exe', '--fail', '--silent', '--show-error', '--location',
                         '--max-redirs', '4', '--max-time', '30', '--max-filesize', str(max(256, size)),
                         '--header', 'Accept-Encoding: identity', '--range', f'{start}-{end}',
                         '--dump-header', str(headers)]
            if self.proxy:
                arguments.extend(['--proxy', self.proxy])
            result = subprocess.run([*arguments, URL], capture_output=True, check=True)
            message = headers.read_text(encoding='iso-8859-1').strip().split('\n\n')[-1].splitlines()
            fields = dict(line.lower().split(': ', 1) for line in message[1:] if ': ' in line)
            expected = f'bytes {start}-{end}/{ARCHIVE_SIZE}'
            if message[0].split()[1] != '206' or fields.get('content-range') != expected:
                raise ValueError('Unexpected archive range')
            if fields.get('etag') != ARCHIVE_TAG:
                raise ValueError('Unexpected archive version')
            data = result.stdout
        if len(data) != size:
            raise ValueError('Bad range length')
        return data


def exposure(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'JPEG' or sorted(image.size) != [3456, 5184]:
            raise ValueError('Unexpected released JPEG geometry')
        main = image.getexif()
        exif = main.get_ifd(34665)
        fields = {'exposure_time': 33434, 'f_number': 33437, 'iso': 34855,
                  'white_balance': 41987, 'exposure_mode': 41986}
        result = {name: float(exif[tag]) if tag in exif else None for name, tag in fields.items()}
        result.update(size=image.size, orientation=main.get(274), camera_model=main.get(272))
        image.verify()
    return result


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def run(out, selection, proxy):
    reader = PublicRange(limit=(34 if selection else 2) * MIB, proxy=proxy)
    out.mkdir()
    with zipfile.ZipFile(reader) as archive:
        entries = [info for info in archive.infolist() if info.filename.lower().endswith('.jpg')]
        index = [{'name': item.filename, 'size': item.file_size, 'crc32': f'{item.CRC:08x}'} for item in entries]
        write(out / 'catalog.json', index)
        photos = {}
        if selection:
            sample = json.loads(selection.read_bytes())
            expected = {f'{subject}-{light}' for subject in ('01', '02', '03') for light in ('warm', 'cold')}
            if expected != set(sample) or len(set(sample.values())) != len(sample):
                raise ValueError('Unexpected sample selection')
            by_name = {item.filename: item for item in entries}
            members = [by_name[name] for name in sample.values()]
            if len(by_name) != len(entries) or sum(item.file_size for item in members) > 32 * MIB:
                raise ValueError('Duplicate entries or sample over budget')
            for key, member in zip(sample, members, strict=True):
                raw = archive.read(member)
                parsed = exposure(raw)
                photos[key] = dict(member=member.filename, sha256=sha256(raw).hexdigest(), bytes=len(raw),
                                   crc32=f'{member.CRC:08x}', exposure=parsed)
                (out / f'{key}.jpg').write_bytes(raw)
    write(out / 'report.json', dict(archive_bytes=ARCHIVE_SIZE, etag=ARCHIVE_TAG,
                                    index_sha256=sha256((out / 'catalog.json').read_bytes()).hexdigest(),
                                    jpeg_members=len(entries), transfer_bytes=reader.transferred, photos=photos))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--proxy')
    run(**vars(parser.parse_args()))
