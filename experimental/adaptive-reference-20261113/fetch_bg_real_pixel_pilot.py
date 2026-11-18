"""Download the fixed, attributed BG-REAL adult-portrait test members."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import requests

from audit_bg_real_controls import REVISION


REPOSITORY = 'https://huggingface.co/datasets/anush-123/bg-real'
LIMIT = 1024**2


def cohort(evidence):
    return [(key,row) for key,row in sorted(evidence['pairs'].items())
            if row['subject'] in ('Man','Woman')][:4]


def fetch(path, expected, proxy, expected_size=None, expected_object=None):
    assert path.startswith(('v2/images/','v2/masks/subject/'))
    assert '..' not in path.split('/')
    proxies = {'https':proxy,'http':proxy} if proxy else None
    url = f'{REPOSITORY}/resolve/{REVISION}/{path}'
    with requests.get(url,proxies=proxies,timeout=(12,25),stream=True) as response:
        response.raise_for_status()
        headers = (response.history or [response])[0].headers
        assert headers.get('X-Repo-Commit')==REVISION
        linked = headers.get('X-Linked-ETag',headers.get('ETag','')).strip('"')
        assert len(linked) in (40,64)
        size = int(response.headers['Content-Length'])
        assert 0<size<=LIMIT
        if (expected_size is not None and size != expected_size
                or expected_object is not None and linked != expected_object):
            raise ValueError('The release object differs from its verified preflight')
        data = response.raw.read(size+1)
    assert len(data)==size
    content_sha = hashlib.sha256(data).hexdigest()
    if len(linked)==64:
        assert content_sha==linked
    else:
        assert hashlib.sha1(f'blob {size}\0'.encode()+data).hexdigest()==linked
    assert expected is None or content_sha==expected
    return data, dict(path=path, size=size, sha256=content_sha, release_object=linked)


def run(evidence, attribution, out, proxy):
    source = json.loads(evidence.read_bytes())
    assert source['revision']==REVISION and source['valid_pairs']==87
    assert hashlib.sha256(attribution.read_bytes()).hexdigest()==source['files']['attribution.csv']['sha256']
    with attribution.open(encoding='utf-8-sig',newline='') as stream:
        credits = {row['bgreal_id']:row for row in csv.DictReader(stream)}
    selected = cohort(source)
    assert len(selected)==4
    out.mkdir(exist_ok=True)
    work = []
    for key,row in selected:
        work.extend((key,name,arm['path'],arm['sha256']) for name,arm in row['arms'].items())
        work.append((key,'mask',row['subject_mask'],None))

    def download(task):
        key, role, remote, expected = task
        data, identity = fetch(remote,expected,proxy)
        filename = f'{key}-{role}{Path(remote).suffix}'
        dest = out/filename
        if dest.exists():
            assert dest.read_bytes()==data
        else:
            with dest.open('xb') as stream:
                stream.write(data)
        print(filename, len(data), flush=True)
        return key,role,dict(**identity,name=filename)

    records = {key:{} for key,_ in selected}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for key,role,data in executor.map(download,work):
            records[key][role] = data
    attributed = {key:[credits[key],credits[row['donor']]] for key,row in selected}
    for name, payload in (('intake.json',records),('attribution.json',attributed)):
        path = out/name
        text = json.dumps(payload,indent=2,ensure_ascii=False)+'\n'
        if path.exists():
            assert path.read_text(encoding='utf-8')==text
        else:
            with path.open('x',encoding='utf-8',newline='\n') as stream:
                stream.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','attribution','out'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    run(args.evidence,args.attribution,args.out,args.proxy)
