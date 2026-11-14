"""Download only the frozen expanded cohort using pinned verified identities."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError,URLError
from urllib.parse import quote
from independent_coco import ROOT,digest,fetch
from coverage_asset_plan import COHORT_SHA,validate_metadata


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'cohort-training-coverage-v1'
    if digest(out/'cohort.json')!=COHORT_SHA:raise ValueError('Changed cohort')
    cohort=json.loads((out/'cohort.json').read_text(encoding='utf-8'));preflight=json.loads((out/'metadata-preflight-v1/results.json').read_text(encoding='utf-8'))
    meta=out/'download-metadata.json';identity=out/'download-identity.json'
    if not preflight['finished'] or digest(meta)!=preflight['metadata_sha256'] or digest(identity)!=preflight['identity_sha256']:raise ValueError('Changed remote preflight')
    for n,sha in preflight['source_sha256'].items():
        if digest(out/'metadata-preflight-v1/frozen-source'/n)!=sha or digest(Path(__file__).with_name(n))!=sha:raise ValueError('Preflight source changed')
    paths={r[k] for part in ('train','validation') for r in cohort['groups'][part] for k in ('content','reference','gt')};records=json.loads(meta.read_text(encoding='utf-8'));total=validate_metadata(records,paths)
    if len(records)!=9162 or total!=preflight['bytes']:raise ValueError('Unexpected asset set/size')
    if shutil.disk_usage(out).free<3*total+2*1024**3:raise ValueError('Insufficient disk')
    names=['download_training_coverage.py','independent_coco.py','coverage_asset_plan.py','training-coverage-duplicates-protocol.md','triplet_color.py'];sources={n:digest(Path(__file__).with_name(n)) for n in names}
    work=out/'download-v1';work.mkdir(exist_ok=True);frozen=work/'frozen-source';frozen.mkdir(exist_ok=True)
    binding=dict(cohort_sha256=COHORT_SHA,preflight_sha256=digest(out/'metadata-preflight-v1/results.json'),source_sha256=sources)
    if (work/'binding.json').exists():
        if json.loads((work/'binding.json').read_text(encoding='utf-8'))!=binding:raise ValueError('Download resume identity changed')
    else:
        for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
        (work/'binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    def verify(path,item):
        if path.stat().st_size!=item['size']:raise ValueError('Asset size differs')
        sha=item.get('lfs',{}).get('oid')
        if sha:
            if digest(path)!=sha:raise ValueError('Asset LFS differs')
        else:
            raw=path.read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['oid']:raise ValueError('Asset Git identity differs')
    def prepare(item):
        path=out/'inputs'/item['path'];root=(out/'inputs').resolve()
        if not path.resolve().is_relative_to(root):raise ValueError('Unsafe asset path')
        reused=False;already=path.exists()
        if not already:
            for name in ('cohort-multiref-v1','cohort-v2','cohort-v1'):
                cached=base/name/'inputs'/item['path']
                if not cached.is_file():continue
                verify(cached,item);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.reuse-partial')
                try:shutil.copyfile(cached,temp);os.replace(temp,path)
                finally:temp.unlink(missing_ok=True)
                reused=True;break
        for attempt in range(3):
            try:
                fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{cohort["revision"]}/{quote(item["path"])}',path,item.get('lfs',{}).get('oid'),item['size']);break
            except (URLError,TimeoutError) as error:
                if attempt==2 or isinstance(error,HTTPError) and error.code not in (429,500,502,503,504):raise
                time.sleep(2**attempt)
        verify(path,item)
        return dict(path=item['path'],sha256=digest(path),bytes=item['size'],reused_previous_bytes=reused,already_present=already)
    print(json.dumps(dict(files=len(records),bytes=total,phase='verified original bytes; no pixel decoding')),flush=True)
    done=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(prepare,item) for item in records]
        for future in as_completed(jobs):
            done.append(future.result())
            if len(done)%100==0:print('verified files',len(done),'/',len(records),flush=True)
    report=dict(binding,metadata_sha256=digest(meta),files=sorted(done,key=lambda v:v['path']),total_bytes=total,finished=True,training_started=False,no_image_pixels_decoded=True)
    (out/'inputs-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,files=len(done),bytes=total,reused=sum(v['reused_previous_bytes'] for v in done),already_present=sum(v['already_present'] for v in done))),flush=True)


if __name__=='__main__':main()
