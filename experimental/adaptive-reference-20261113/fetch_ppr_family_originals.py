"""Bounded original acquisition; no pixel decoding or external publication."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
from urllib.parse import quote
from independent_coco import ROOT,digest

OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-family-originals-v1'
CACHE=ROOT/'.tools/PPR10K-audit/family-originals'

def verify(path,row):
    raw=path.read_bytes();assert len(raw)==row['size']
    actual=hashlib.sha256(raw).hexdigest()
    if 'lfs' in row:assert actual==row['lfs']['oid']
    else:assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==row['oid']
    return actual

def main():
    plan=json.loads((OUT/'plan.json').read_text());assert plan['finished'] and plan['total_bytes']<=256*1024**2
    assert digest(OUT/'plan.json')=='8aa1de9f34a8d3fe4f1bcaadb4df2b86ca83504826fcacae44007b1bcab26729'
    report=dict(plan_sha256=digest(OUT/'plan.json'),driver_sha256=digest(__file__),files=[],finished=False,pixel_decoding=False)
    status=OUT/'download-progress.json';assert not (OUT/'download-results.json').exists()
    CACHE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    old_roots=sorted(list(base.glob('*/inputs'))+list(base.glob('*/originals')))
    def fetch(row):
        assert time.monotonic()-start<600
        p=CACHE/row['path'];assert p.resolve().is_relative_to(CACHE.resolve())
        p.parent.mkdir(parents=True,exist_ok=True);reused=False
        if p.exists():sha=verify(p,row);reused=True
        else:
            old=None
            for root in old_roots:
                candidate=root/row['path']
                if candidate.is_file() and candidate.stat().st_size==row['size']:
                    try:verify(candidate,row);old=candidate;break
                    except AssertionError:continue
            if old is not None:
                # Keep the verified existing file in place; records point at it.
                p=old;sha=verify(p,row);reused=True
            else:
                partial=p.with_suffix(p.suffix+'.partial');assert not partial.exists(),'Inspect previous partial before retry'
                op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
                url=f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{plan["revision"]}/{quote(row["path"])}?download=true'
                count=0
                with op.open(url,timeout=45) as response,partial.open('wb') as stream:
                    while block:=response.read(1024*1024):
                        count+=len(block);assert count<=row['size'] and time.monotonic()-start<600
                        stream.write(block)
                sha=verify(partial,row);partial.rename(p)
        return dict(path=row['path'],local_path=str(p.relative_to(ROOT)),bytes=row['size'],sha256=sha,reused=reused)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(fetch,row) for row in plan['files']]
        for future in concurrent.futures.as_completed(pending):
            report['files'].append(future.result())
            status.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if len(report['files'])%20==0:print(json.dumps(dict(verified=len(report['files']),total=159)),flush=True)
    assert len(report['files'])==159;report['files'].sort(key=lambda r:r['path'])
    report.update(finished=True,seconds=time.monotonic()-start,new_bytes=sum(r['bytes'] for r in report['files'] if not r['reused']),
                  reused_files=sum(r['reused'] for r in report['files']))
    assert digest(__file__)==report['driver_sha256']
    (OUT/'download-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['finished','seconds','new_bytes','reused_files']}),flush=True)

if __name__=='__main__':main()
