"""Read the tar index of two fixed SyncLight real scenes via verified ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import requests


REVISION = '8067588dc4ee901b805c3813d234a661973f8e71'
REPO = 'https://huggingface.co/datasets/davidserra9/synclight'
class ArchiveHeaderReader:
    def __init__(self, release, proxy):
        self.row = release
        self.cursor = 0
        self.responses = []
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update({'http': proxy, 'https': proxy})

    def tell(self):
        return self.cursor

    def seek(self, where, whence=io.SEEK_SET):
        assert whence == io.SEEK_SET and 0 <= where <= self.row['bytes']
        self.cursor = where

    def read(self, count):
        if count == 0:
            return b''
        if count < 0 or count > 4096 or len(self.responses) >= 80:
            raise ValueError('Only bounded tar header reads are permitted')
        start, stop = self.cursor, self.cursor+count-1
        url = f"{REPO}/resolve/{REVISION}/{self.row['path']}"
        with self.session.get(url, headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'},
                              timeout=(10, 15), stream=True) as response:
            source = (response.history or [response])[0].headers
            if source.get('X-Repo-Commit') != REVISION or source.get('X-Linked-Etag', '').strip('"') != self.row['sha256']:
                raise ValueError('Archive revision or hash differs')
            if response.status_code != 206 or response.headers.get('Content-Range') != (
                f"bytes {start}-{stop}/{self.row['bytes']}"
            ) or int(response.headers.get('Content-Length', -1)) != count:
                raise ValueError(f'Expected range {start}-{stop}, got {response.status_code} {response.headers.get("Content-Range")}')
            data = response.raw.read(count+1)
        if len(data) != count:
            raise ValueError('Short archive range')
        self.responses.append(dict(start=start, size=count, sha256=sha256(data).hexdigest()))
        self.cursor += count
        return data


def archive_index(row, proxy):
    reader = ArchiveHeaderReader(row, proxy)
    try:
        with tarfile.open(fileobj=reader, mode='r:') as archive:
            entries = []
            for member in archive:
                if member.isfile():
                    if Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                        raise ValueError('Unexpected archive member path')
                    entries.append(dict(path=member.name, start=member.offset_data, size=member.size))
    finally:
        reader.session.close()
    raw = [item for item in entries if Path(item['path']).suffix.lower() in ('.dng', '.cr2')]
    assert len(raw) == row['n_raw'] and len(entries)-len(raw) == row['n_light_map']
    return dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'],
                file_members=entries, requests=reader.responses)


def run(source, out, proxy, split):
    meta = json.loads((source/'meta.json').read_bytes())
    tree = json.loads((source/f'{split}.json').read_bytes())
    manifest = [json.loads(line) for line in (source/'manifest.jsonl').read_bytes().splitlines()]
    assert meta['sha'] == REVISION and meta['cardData']['license'] == 'cc-by-4.0' and meta['gated'] is False
    splits = {entry['scene']: entry for entry in manifest if entry['source']=='real' and entry['split']==split}
    assert len(splits) == len(tree)
    for item in tree:
        scene = Path(item['path']).stem
        row = splits[scene]
        assert (item['path'], item['size'], item['lfs']['oid']) == (row['path'], row['bytes'], row['sha256'])
    if split == 'test':
        rows = [splits[name] for name in ('scene15', 'scene16')]
    else:
        rows = sorted((entry for entry in splits.values() if entry['n_raw']>=4),
                      key=lambda row:(row['bytes'], row['scene']))[:2]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda row: archive_index(row, proxy), rows))
    counts = sum(req['size'] for scene in results for req in scene['requests'])
    incoming = {file.name: file.read_bytes() for file in source.iterdir() if file.is_file()}
    assert sum(map(len, incoming.values()))+counts <= 1024**2
    payload = dict(revision=REVISION, total_header_bytes=counts,
                   scenes={row['scene']: result for row, result in zip(rows, results, strict=True)},
                   metadata={name: sha256(data).hexdigest() for name, data in sorted(incoming.items())},
                   metadata_bytes=sum(map(len, incoming.values())))
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, indent=2)+'\n')
    for scene in results:
        print(scene['path'], scene['file_members'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    parser.add_argument('--split', choices=('test','train'), default='test')
    args = parser.parse_args()
    run(args.source, args.out, args.proxy, args.split)
