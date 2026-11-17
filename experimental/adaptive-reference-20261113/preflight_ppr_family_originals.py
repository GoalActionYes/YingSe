"""Pinned bounded metadata plan for already-consumed reference families."""
import json,re,time,urllib.request
from pathlib import Path,PurePosixPath
from independent_coco import ROOT,digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
PARENT=BASE/'ppr-group-exclusions-v2/results.json'
OUT=BASE/'ppr-family-originals-v1'
REV='63bf13833b75407c862d608c3d81281d0a35b61f'

def main():
    OUT.mkdir(exist_ok=False);parent=json.loads(PARENT.read_text());assert parent['finished']
    paths=sorted(q['path'] for q in parent['queries'] if q['status']=='missing_registered_identity');assert len(paths)==159
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}));records=[]
    for offset in range(0,len(paths),100):
        chunk=paths[offset:offset+100]
        req=urllib.request.Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{REV}',
            data=json.dumps(dict(paths=chunk)).encode(),headers={'Content-Type':'application/json'})
        with op.open(req,timeout=30) as r:rows=json.load(r)
        assert {v['path'] for v in rows}==set(chunk);records.extend(rows)
    assert len(records)==159 and {v['path'] for v in records}==set(paths)
    for v in records:
        p=PurePosixPath(v['path']);assert not p.is_absolute() and '..' not in p.parts and str(p)==v['path']
        assert v['type']=='file' and 0<v['size']<=20*1024**2 and re.fullmatch('[0-9a-f]{40}',v['oid'])
        if 'lfs' in v:assert re.fullmatch('[0-9a-f]{64}',v['lfs']['oid']) and v['lfs']['size']==v['size']
    total=sum(v['size'] for v in records);assert total<=256*1024**2
    card=ROOT/'.tools/TST100K-index/README.md'
    assert digest(card)=='a811aa7dfb47685fbfd1b00504abda17edf5ecf163bafd75ebc9f96d19d79719'
    plan=dict(scope='original files of consumed reference families only; no target GT or new evaluation',revision=REV,
        parent_sha256=digest(PARENT),dataset_card_sha256=digest(card),source_sha256=digest(__file__),
        protocol_sha256=digest(Path(__file__).with_name('ppr-family-originals-protocol.md')),
        total_bytes=total,files=sorted(records,key=lambda v:v['path']),metadata_only=True,finished=True)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (OUT/'protocol.md').write_bytes(Path(__file__).with_name('ppr-family-originals-protocol.md').read_bytes())
    print(json.dumps(dict(files=159,total_bytes=total,plan_sha256=digest(OUT/'plan.json'))))

if __name__=='__main__':main()
