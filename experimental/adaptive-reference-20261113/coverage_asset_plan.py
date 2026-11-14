"""Validate bounded pinned asset metadata without downloading image bytes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from independent_coco import ROOT,digest
from prepare_training_coverage import EXPECTED

COHORT_SHA='d001cec5f2f3386e6237627524b88ea42da70feb7bb8d49a60281e1df434aeb5'
MAX_BYTES=8*1024**3


def validate_metadata(records,expected,max_bytes=MAX_BYTES):
    seen=set();total=0
    for item in records:
        path=item['path'];size=item['size']
        if not isinstance(path,str) or not path or '\\' in path or ':' in path or str(PurePosixPath(path))!=path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            raise ValueError('Unsafe metadata path')
        if path in seen:raise ValueError('Duplicate metadata path')
        seen.add(path)
        if type(size) is not int or not 0<size<=20*1024**2:raise ValueError('Invalid asset byte size')
        if not re.fullmatch('[0-9a-f]{40}',item.get('oid','')):raise ValueError('Invalid Git identity')
        if 'lfs' in item and not re.fullmatch('[0-9a-f]{64}',item['lfs'].get('oid','')):raise ValueError('Invalid LFS identity')
        total+=size
    if seen!=set(expected):raise ValueError('Asset metadata set differs')
    if total>max_bytes:raise ValueError('Frozen byte bound exceeded')
    return total


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'cohort-training-coverage-v1';cohort_path=out/'cohort.json'
    if digest(cohort_path)!=COHORT_SHA:raise ValueError('Expanded metadata changed')
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));revision=cohort['revision']
    paths={r[k] for part in ('train','validation') for r in cohort['groups'][part] for k in ('content','reference','gt')}
    if len(paths)!=9162 or cohort['groups']['reserved']:raise ValueError('Expanded path count changed')
    cache={};cache_hashes={}
    for name in ('cohort-v1','cohort-v2','cohort-multiref-v1'):
        old=base/name
        if not (old/'download-identity.json').is_file():continue
        identity=json.loads((old/'download-identity.json').read_text(encoding='utf-8'));meta=old/'download-metadata.json'
        if digest(old/'cohort.json')!=EXPECTED[name] or identity['cohort_sha256']!=EXPECTED[name] or identity['revision']!=revision or identity['metadata_sha256']!=digest(meta):
            raise ValueError('Reusable remote metadata identity differs')
        data=json.loads(meta.read_text(encoding='utf-8'));validate_metadata(data,{r['path'] for r in data})
        cache_hashes[name]=digest(meta)
        for item in data:
            if item['path'] in cache and cache[item['path']]!=item:raise ValueError('Reusable metadata disagree')
            cache[item['path']]=item
    known={p:cache[p] for p in paths&cache.keys()};missing=sorted(paths-known.keys())
    work=out/'metadata-preflight-v1';work.mkdir(exist_ok=True);frozen=work/'frozen-source';frozen.mkdir(exist_ok=True)
    names=['coverage_asset_plan.py','training-coverage-bytes-protocol.md','prepare_training_coverage.py']
    source={n:digest(Path(__file__).with_name(n)) for n in names}
    identity=dict(cohort_sha256=COHORT_SHA,revision=revision,source_sha256=source,reused_metadata_sha256=cache_hashes,maximum_total_bytes=MAX_BYTES)
    identity_path=work/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8'))!=identity:raise ValueError('Preflight resume identity changed')
        if any(digest(frozen/n)!=sha for n,sha in source.items()):raise ValueError('Frozen preflight source changed')
    else:
        for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
        identity_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    def request_chunk(number,chunk):
        path=work/f'part-{number:03d}.json'
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'))
            if result['requested']!=chunk:raise ValueError('Saved chunk request differs')
            validate_metadata(result['records'],chunk);return result['records']
        req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{revision}',data=json.dumps(dict(paths=chunk)).encode(),headers={'Content-Type':'application/json'})
        for attempt in range(3):
            try:
                with urlopen(req,timeout=60) as response:data=json.load(response)
                break
            except (URLError,TimeoutError) as error:
                if attempt==2 or isinstance(error,HTTPError) and error.code not in (429,500,502,503,504):raise
                time.sleep(2**attempt)
        validate_metadata(data,chunk);temp=path.with_suffix('.partial');temp.write_text(json.dumps(dict(requested=chunk,records=data),indent=2),encoding='utf-8');temp.replace(path)
        return data
    print(json.dumps(dict(metadata_known=len(known),metadata_missing=len(missing),image_bytes_downloaded=0)),flush=True)
    records=list(known.values());chunks=[missing[i:i+100] for i in range(0,len(missing),100)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(request_chunk,i,chunk) for i,chunk in enumerate(chunks)]
        for done,future in enumerate(as_completed(jobs),1):
            records.extend(future.result())
            if done%10==0 or done==len(jobs):print('metadata batches verified',done,'/',len(jobs),flush=True)
    records.sort(key=lambda r:r['path']);total=validate_metadata(records,paths);free=shutil.disk_usage(out).free
    if free<3*total+2*1024**3:raise ValueError('Insufficient disk for original/prepared copies')
    metadata_path=out/'download-metadata.json';download_identity=out/'download-identity.json'
    metadata_path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    download_identity.write_text(json.dumps(dict(revision=revision,cohort_sha256=COHORT_SHA,metadata_sha256=digest(metadata_path)),indent=2),encoding='utf-8')
    summary=dict(identity,files=len(records),bytes=total,metadata_known=len(known),metadata_new=len(missing),
                 metadata_sha256=digest(metadata_path),identity_sha256=digest(download_identity),finished=True,
                 no_image_bytes_downloaded_or_read=True,training_started=False,app_promoted=False)
    (work/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
