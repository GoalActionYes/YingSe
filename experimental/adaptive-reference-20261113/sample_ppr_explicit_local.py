"""Fetch all three metadata-qualified triples under the existing 16 MiB range cap."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
from sample_ppr_other_experts import SmallZip
from sample_ppr_remote_zip import ASSETS,ROOT,digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-explicit-local-pixels-v1'


def main():
    OUT.mkdir(exist_ok=False)
    qualified=BASE/'ppr-local-effect-structure-v2/results.json'
    assert digest(qualified)=='5531b832612b97c6a6b2237a7c3c4059e026070f5c6195d0fea6e4825b8c95b5'
    data=json.loads(qualified.read_text());ids=data['eligible_for_pixel_check'];assert ids==['244_4','1369_1','1645_1']
    sources={n:digest(Path(__file__).with_name(n)) for n in ['sample_ppr_explicit_local.py','sample_ppr_other_experts.py','sample_ppr_remote_zip.py','ppr-explicit-local-pixel-protocol.md']}
    plan=dict(ids=ids,qualified_sha256=digest(qualified),source_sha256=sources,archives=ASSETS,budget_bytes=16*1024**2)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in sources:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(plan_sha256=digest(OUT/'plan.json'),files=[],requests=[],finished=False,pixels_decoded=False)
    budget=dict(start=time.monotonic(),bytes=0,requests=report['requests'])
    try:
        for role,(file_id,size,suffix) in ASSETS.items():
            remote=SmallZip(role,file_id,size,budget)
            with zipfile.ZipFile(remote) as z:
                infos=z.infolist()
                (OUT/(role+'-index.json')).write_text(json.dumps([dict(name=i.filename,bytes=i.file_size,crc=i.CRC) for i in infos if not i.is_dir()],indent=2),encoding='utf-8')
                for key in ids:
                    matches=[i for i in infos if Path(i.filename).stem==key and Path(i.filename).suffix.lower() in ([suffix,'.tiff'] if suffix=='.tif' else [suffix])]
                    assert len(matches)==1
                    info=matches[0];assert info.file_size<=4*1024**2 and not info.flag_bits&1
                    raw=z.read(info);folder=OUT/'samples'/role;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(key+suffix);path.write_bytes(raw)
                    report['files'].append(dict(role=role,id=key,member=info.filename,bytes=len(raw),crc=info.CRC,sha256=hashlib.sha256(raw).hexdigest(),local_path=str(path.relative_to(ROOT))))
            print(json.dumps(dict(role=role,files=len(report['files']),read_bytes=budget['bytes'])),flush=True)
        report['finished']=True
    finally:
        report.update(read_bytes=budget['bytes'],seconds=time.monotonic()-budget['start'])
        (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert len(report['files'])==9 and all(digest(Path(__file__).with_name(n))==h for n,h in sources.items())
    print(json.dumps(dict(finished=True,read_bytes=budget['bytes'],requests=len(report['requests']))))


if __name__=='__main__':main()
