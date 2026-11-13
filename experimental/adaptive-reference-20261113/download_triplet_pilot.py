"""Fetch only the frozen academic train/validation pilot; never reserved images."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
from independent_coco import ROOT,digest,fetch


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/cohort-v1';cohort=json.loads((out/'cohort.json').read_text(encoding='utf-8'))
    revision=cohort['revision'];paths=sorted({row[role] for part in ['train','validation'] for row in cohort['groups'][part] for role in ['content','reference','gt']})
    reserved={row[role] for row in cohort['groups']['reserved'] for role in ['content','reference','gt']}
    if set(paths)&reserved:raise ValueError('Reserved input included')
    metadata=[]
    for start in range(0,len(paths),100):
        req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{revision}',data=json.dumps({'paths':paths[start:start+100]}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=60) as response:metadata.extend(json.load(response))
    if {x['path'] for x in metadata}!=set(paths):raise ValueError('Asset set mismatch')
    total=sum(x['size'] for x in metadata)
    if total>600*1024*1024:raise ValueError('Pilot exceeds bounded 600MiB download')
    (out/'download-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),bytes=total,train=128,validation=32,reserved_downloaded=False)),flush=True)
    def prepare(item):
        path=out/'inputs'/item['path']
        if not path.resolve().is_relative_to((out/'inputs').resolve()):raise ValueError('Unsafe input path')
        sha=item.get('lfs',{}).get('oid')
        fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{revision}/{quote(item["path"])}',path,sha,item['size'])
        if not sha:
            raw=path.read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['oid']:raise ValueError('Git blob mismatch')
        return dict(path=item['path'],sha256=digest(path),bytes=item['size'])
    done=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(prepare,metadata):
            done.append(item)
            if len(done)%40==0:print('verified files',len(done),flush=True)
    report=dict(cohort_sha256=digest(out/'cohort.json'),source_sha256=digest(Path(__file__)),files=done,total_bytes=total,
        scope='Academic train and validation pilot only',reserved_images_downloaded=False,training_started=False)
    (out/'inputs-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('pilot complete',len(done),flush=True)


if __name__=='__main__':main()
