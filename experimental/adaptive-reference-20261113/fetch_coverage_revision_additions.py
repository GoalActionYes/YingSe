"""Fetch only the nine frozen v2 additions; reuse v1 metadata without mutation."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
from independent_coco import ROOT,digest,fetch
from coverage_asset_plan import validate_metadata


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';v1=base/'cohort-training-coverage-v1';v2=base/'cohort-training-coverage-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=digest(v2/'cohort.json')
    if sha!='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077':raise ValueError('Revision changed')
    c=load(v2/'cohort.json');audit=load(v2/'alias-evidence-audit-v1/results.json')
    if not audit['passed'] or audit['cohort_sha256']!=sha:raise ValueError('Alias evidence not accepted')
    old_meta=v1/'download-metadata.json'
    if digest(old_meta)!=load(v1/'metadata-preflight-v1/results.json')['metadata_sha256']:raise ValueError('Old remote identities changed')
    wanted=c['new_asset_paths']
    if len(wanted)!=9:raise ValueError('Addition count differs')
    out=v2/'additions-v1';out.mkdir(exist_ok=True);meta=out/'metadata.json';identity=out/'identity.json'
    source=digest(Path(__file__))
    if meta.exists():
        records=load(meta);binding=load(identity)
        if binding!=dict(cohort_sha256=sha,source_sha256=source,metadata_sha256=digest(meta),alias_audit_sha256=digest(v2/'alias-evidence-audit-v1/results.json')):raise ValueError('Addition resume changed')
    else:
        req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{c["revision"]}',data=json.dumps(dict(paths=wanted)).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=60) as response:records=json.load(response)
        validate_metadata(records,wanted);meta.write_text(json.dumps(records,indent=2),encoding='utf-8');binding=dict(cohort_sha256=sha,source_sha256=source,metadata_sha256=digest(meta),alias_audit_sha256=digest(v2/'alias-evidence-audit-v1/results.json'));identity.write_text(json.dumps(binding,indent=2),encoding='utf-8');(out/'frozen-source.py').write_bytes(Path(__file__).read_bytes())
    validate_metadata(records,wanted)
    paths={r[k] for part in ('train','validation') for r in c['groups'][part] for k in ('content','reference','gt')};old={r['path']:r for r in load(old_meta)};combined={**old,**{r['path']:r for r in records}};all_meta=[combined[p] for p in sorted(paths)];total=validate_metadata(all_meta,paths)
    done=[]
    for r in records:
        p=v2/'inputs'/r['path']
        for attempt in range(3):
            try:fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{c["revision"]}/{quote(r["path"])}',p,r.get('lfs',{}).get('oid'),r['size']);break
            except (TimeoutError,ValueError) as error:
                if attempt==2 or isinstance(error,ValueError) and str(error)!='Incomplete download':raise
        if 'lfs' not in r:
            data=p.read_bytes()
            if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=r['oid']:raise ValueError('Git identity differs')
        done.append(dict(path=r['path'],sha256=digest(p),bytes=p.stat().st_size))
    if digest(v2/'cohort.json')!=sha or digest(Path(__file__))!=source:raise ValueError('Addition evidence changed')
    (v2/'download-metadata.json').write_text(json.dumps(all_meta,indent=2),encoding='utf-8');report=dict(binding,files=done,finished=True,total_revision_bytes=total,added_bytes=sum(v['bytes'] for v in done),new_pixels_decoded=False,training_started=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,new_files=len(done),added_bytes=report['added_bytes'],total_revision_bytes=total,new_pixels_decoded=False)),flush=True)


if __name__=='__main__':main()
