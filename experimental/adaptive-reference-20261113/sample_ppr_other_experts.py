"""Fetch only eight fixed expert B/C labels with a stricter inherited range budget."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
from sample_ppr_remote_zip import RemoteZip, ROOT, digest

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-expert-contrast-v1'
ARCHIVES={'b':('1AqH2smHSDT7Ba7fLSzbosEANOMi3BqzK',4549796317),'c':('1U3vRtu7QJR81eOonmArQ0QDGgW-DUKIY',4725705345)}


class SmallZip(RemoteZip):
    def read(self,n=-1):
        amount=self.size-self.position if n<0 else min(n,self.size-self.position)
        assert self.budget['bytes']+amount<=16*1024**2 and len(self.budget['requests'])<64
        return super().read(n)


def main():
    OUT.mkdir(exist_ok=False)
    original=BASE/'ppr-pixel-sample-v1/plan.json';previous=BASE/'ppr-source-pixel-link-v1/results.json'
    linkage=json.loads(previous.read_text());assert linkage['finished'] and all(r['same_file_bytes'] for r in linkage['rows'])
    assert [r['id'] for r in linkage['rows']]==json.loads(original.read_text())['ids']
    ids=json.loads(original.read_text())['ids']
    sources={n:digest(Path(__file__).with_name(n)) for n in ['sample_ppr_other_experts.py','sample_ppr_remote_zip.py','ppr-expert-contrast-protocol.md']}
    plan=dict(ids=ids,source_link_sha256=digest(previous),original_plan_sha256=digest(original),source_sha256=sources,archives=ARCHIVES,budget_bytes=16*1024**2)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in sources:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(plan_sha256=digest(OUT/'plan.json'),files=[],requests=[],finished=False,pixels_decoded=False)
    budget=dict(start=time.monotonic(),bytes=0,requests=report['requests'])
    try:
        for expert,(file_id,size) in ARCHIVES.items():
            remote=SmallZip(expert,file_id,size,budget)
            with zipfile.ZipFile(remote) as z:
                infos=z.infolist()
                index=[dict(name=i.filename,bytes=i.file_size,crc=i.CRC) for i in infos if not i.is_dir()]
                (OUT/(expert+'-index.json')).write_text(json.dumps(index,indent=2),encoding='utf-8')
                for key in ids:
                    matches=[i for i in infos if Path(i.filename).stem==key and Path(i.filename).suffix.lower() in ['.tif','.tiff']]
                    assert len(matches)==1
                    info=matches[0];assert info.file_size<=4*1024**2 and not info.flag_bits&1
                    raw=z.read(info);folder=OUT/'samples'/expert;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(key+'.tif');path.write_bytes(raw)
                    report['files'].append(dict(expert=expert,id=key,member=info.filename,bytes=len(raw),crc=info.CRC,sha256=hashlib.sha256(raw).hexdigest(),local_path=str(path.relative_to(ROOT))))
            print(json.dumps(dict(expert=expert,files=len(report['files']),read_bytes=budget['bytes'])),flush=True)
        report['finished']=True
    finally:
        report.update(read_bytes=budget['bytes'],seconds=time.monotonic()-budget['start'])
        (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert len(report['files'])==8
    assert all(digest(Path(__file__).with_name(n))==h for n,h in sources.items())
    print(json.dumps(dict(finished=True,ids=ids,read_bytes=budget['bytes'],requests=len(report['requests']))))


if __name__=='__main__':main()
