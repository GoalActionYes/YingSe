"""Read the official FAU repainting archive index under a strict range budget."""
import argparse
from collections import Counter
from ftplib import FTP
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.request import Request, urlopen
from zipfile import ZipFile


URL = ('https://www5.informatik.uni-erlangen.de/fileadmin/research/datasets/'
       'color_and_reflectance/cpcv_2011_data.zip')
BUDGET = 512 * 1024


class RemoteIndex:
    def __init__(self):
        with urlopen(Request(URL, method='HEAD'), timeout=15) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers['ETag']
            assert response.headers['Accept-Ranges'] == 'bytes' and self.etag
        self.offset = 0
        self.requests = []

    def tell(self):
        return self.offset

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, self.size)[whence] + offset
        if self.offset < 0:
            raise OSError('Seek starts before archive')
        return self.offset

    def read(self, length=-1):
        available = max(0, self.size-self.offset)
        length = available if length < 0 else min(length, available)
        if not length:
            return b''
        if length + sum(request['bytes'] for request in self.requests) > BUDGET:
            raise OSError('Index range exceeds download budget')
        first, last = self.offset, self.offset+length-1
        headers = {'Range': f'bytes={first}-{last}', 'If-Match': self.etag,
                   'Accept-Encoding': 'identity'}
        with urlopen(Request(URL, headers=headers), timeout=15) as response:
            if (response.status != 206
                or response.headers.get('Content-Range') != f'bytes {first}-{last}/{self.size}'
                or response.headers.get('ETag') != self.etag):
                raise OSError('Archive byte range or identity did not match')
            data = response.read(length+1)
        if len(data) != length:
            raise OSError('Archive byte range has unexpected length')
        self.offset += length
        self.requests.append(dict(first=first, last=last, bytes=length, sha256=hashlib.sha256(data).hexdigest()))
        return data


def run(out):
    archive = RemoteIndex()
    with ZipFile(archive) as zipped:
        files = zipped.infolist()
        readme = zipped.getinfo('cpcv_2011_data/README.txt')
        assert readme.file_size <= 4096 and readme.compress_size <= 4096
        description = zipped.read(readme)
    names = [Path(entry.filename) for entry in files if not entry.is_dir()]
    prefix_counts = Counter('/'.join(name.parts[:2]) for name in names)
    record = dict(url=URL, bytes=archive.size, etag=archive.etag, requests=archive.requests,
                  index_entries=len(files), non_directory=len(names), prefix_counts=prefix_counts,
                  extensions=Counter(path.suffix.lower() for path in names),
                  readme_sha256=hashlib.sha256(description).hexdigest(),
                  readme=description.decode('utf-8', errors='replace'),
                  text_entries=[path.as_posix() for path in names if not re.match(r'\.(png|jpg|tif)$', path.suffix, re.I)],
                  members=[name.as_posix() for name in names])
    with out.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def mls_index(out):
    host = 'vis.iitp.ru'
    name = '/mls-dataset/images_preview.zip'
    with FTP(host, timeout=15) as ftp:
        ftp.login()
        ftp.voidcmd('TYPE I')
        size = ftp.size(name)
        assert size == 93125444

        def tail(offset, length):
            assert length <= BUDGET and offset+length == size
            with ftp.transfercmd(f'RETR {name}', rest=offset) as connection:
                with connection.makefile('rb') as stream:
                    payload = stream.read(length+1)
            ftp.voidresp()
            assert len(payload) == length
            return payload

        eocd = tail(size-22, 22)
        sig, disk, start_disk, disk_count, count, length, offset, comment = struct.unpack('<4s4H2LH', eocd)
        assert sig == b'PK\x05\x06' and disk == start_disk == comment == 0 and disk_count == count
        payload = tail(offset, length+22)
        assert payload[-22:] == eocd

    with ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
    assert len(entries) == count
    members = [PurePosixPath(entry.filename) for entry in entries if not entry.is_dir()]
    scenes = Counter(path.parent.name for path in members)
    assert len(scenes) == 24 and set(scenes.values()) == {18}
    settings = {scene: sorted(path.stem.removeprefix(scene+'_') for path in members
                              if path.parent.name == scene) for scene in scenes}
    assert all(value == settings['01'] for value in settings.values())
    with out.open('x', encoding='utf-8') as stream:
        json.dump(dict(url=f'ftp://{host}{name}', archive_bytes=size, requests_bytes=len(payload)+len(eocd),
                       central_and_eocd_sha256=hashlib.sha256(payload).hexdigest(), images=len(members),
                       scenes=dict(sorted(scenes.items())), settings=settings['01']), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--mls', action='store_true')
    args = parser.parse_args()
    (mls_index if args.mls else run)(args.out)
