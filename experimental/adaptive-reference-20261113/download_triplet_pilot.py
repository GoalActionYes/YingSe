"""Fetch only the frozen academic train/validation pilot; never reserved images."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from urllib.error import URLError,HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen
from independent_coco import ROOT,digest,fetch
from triplet_dataset_options import dataset_options


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--expanded',action='store_true');mode.add_argument('--multiref',action='store_true');args=parser.parse_args()
    options=dataset_options(ROOT,args.expanded,args.multiref);out=options['directory'];cohort=json.loads((out/'cohort.json').read_text(encoding='utf-8'))
    revision=cohort['revision'];paths=sorted({row[role] for part in ['train','validation'] for row in cohort['groups'][part] for role in ['content','reference','gt']})
    reserved={row[role] for row in cohort['groups']['reserved'] for role in ['content','reference','gt']}
    if set(paths)&reserved:raise ValueError('Reserved input included')
    metadata=[];metadata_path=out/'download-metadata.json';identity_path=out/'download-identity.json'
    if identity_path.is_file():
        identity=json.loads(identity_path.read_text(encoding='utf-8'))
        if identity['revision']!=revision or identity['cohort_sha256']!=digest(out/'cohort.json') or identity['metadata_sha256']!=digest(metadata_path):raise ValueError('Changed metadata snapshot')
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    else:
        for start in range(0,len(paths),100):
            req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{revision}',data=json.dumps({'paths':paths[start:start+100]}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=60) as response:metadata.extend(json.load(response))
    if {x['path'] for x in metadata}!=set(paths) or len(metadata)!=len(paths):raise ValueError('Asset set mismatch')
    total=sum(x['size'] for x in metadata)
    if total>options['max_bytes'] or any(x['size']>20*1024*1024 for x in metadata):raise ValueError('Pilot exceeds frozen download bounds')
    if not identity_path.is_file():
        metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        identity_path.write_text(json.dumps(dict(revision=revision,cohort_sha256=digest(out/'cohort.json'),metadata_sha256=digest(metadata_path)),indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),bytes=total,train=len(cohort['groups']['train']),validation=len(cohort['groups']['validation']),reserved_downloaded=False)),flush=True)
    def prepare(item):
        path=out/'inputs'/item['path']
        if not path.resolve().is_relative_to((out/'inputs').resolve()):raise ValueError('Unsafe input path')
        sha=item.get('lfs',{}).get('oid');reused=False
        for previous in options['reuse']:
            cached=previous/'inputs'/item['path']
            if path.exists() or not cached.is_file():continue
            if cached.stat().st_size!=item['size'] or (sha and digest(cached)!=sha):raise ValueError('Changed reusable asset')
            if not sha:
                raw=cached.read_bytes()
                if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['oid']:raise ValueError('Reusable Git blob mismatch')
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.reuse-partial')
            try:shutil.copyfile(cached,temp);os.replace(temp,path)
            finally:temp.unlink(missing_ok=True)
            reused=True
            break
        for attempt in range(3):
            try:
                fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{revision}/{quote(item["path"])}',path,sha,item['size']);break
            except (URLError,TimeoutError) as error:
                if attempt==2 or isinstance(error,HTTPError) and error.code not in (429,500,502,503,504):raise
                time.sleep(2**attempt)
        if not sha:
            raw=path.read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['oid']:raise ValueError('Git blob mismatch')
        return dict(path=item['path'],sha256=digest(path),bytes=item['size'],reused_previous_bytes=reused)
    done=[];pool=ThreadPoolExecutor(max_workers=4);futures=[pool.submit(prepare,item) for item in metadata]
    try:
        for future in as_completed(futures):
            item=future.result()
            done.append(item)
            if len(done)%40==0:print('verified files',len(done),flush=True)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    report=dict(cohort_sha256=digest(out/'cohort.json'),source_sha256=digest(Path(__file__)),options_source_sha256=digest(Path(__file__).with_name('triplet_dataset_options.py')),files=sorted(done,key=lambda x:x['path']),total_bytes=total,
        scope='Academic train and validation pilot only',reserved_images_downloaded=False,training_started=False)
    (out/'inputs-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('pilot complete',len(done),flush=True)


if __name__=='__main__':main()
