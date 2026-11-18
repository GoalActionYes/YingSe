"""Fetch the eight pinned RAW members without downloading the archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re

import requests

from audit_synclight_real_intake import REPO, REVISION


def fetch(scene, member, directory, proxy):
    start = member['start']
    size = member['size']
    stop = start+size-1
    name = Path(member['path']).name
    dest = directory/name
    assert not dest.exists()
    proxies = {'http': proxy, 'https': proxy} if proxy else None
    url = f"{REPO}/resolve/{REVISION}/{scene['path']}"
    with requests.get(url, headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'},
                      proxies=proxies, timeout=(12, 30), stream=True) as response:
        source = (response.history or [response])[0].headers
        assert source['X-Repo-Commit'] == REVISION and source['X-Linked-Etag'].strip('"') == scene['sha256']
        assert response.status_code == 206
        assert response.headers['Content-Range'] == f"bytes {start}-{stop}/{scene['bytes']}"
        assert int(response.headers['Content-Length']) == size
        data = response.raw.read(size+1)
    assert len(data) == size and data[:4] in (b'II*\0', b'MM\0*')
    with dest.open('xb') as stream:
        stream.write(data)
    print(name, len(data), flush=True)
    return name, dict(member=member['path'], sha256=sha256(data).hexdigest(),
                      size=size, offset=start, archive_sha256=scene['sha256'])


def run(evidence, out, proxy):
    data = json.loads(evidence.read_bytes())
    assert data['train']['revision'] == REVISION
    all_scenes = data['train']['scenes']
    assert list(all_scenes) == ['scene35', 'scene25']
    tasks = []
    for name, scene in all_scenes.items():
        matches = [m for m in scene['file_members'] if re.fullmatch(
            rf'{name}/captured/{name}_[12]_(?:0_ambient|1_camvis)\.DNG', m['path'])]
        assert len(matches) == 4
        tasks.extend((scene, member) for member in matches)
    size = sum(member['size'] for _, member in tasks)
    assert size == 106973902
    out.mkdir()
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = dict(pool.map(lambda pair: fetch(*pair, out, proxy), tasks))
    assert len(files) == 8
    record = dict(revision=REVISION, source_sha256=sha256(evidence.read_bytes()).hexdigest(), files=files)
    (out/'intake.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    run(args.evidence, args.out, args.proxy)
