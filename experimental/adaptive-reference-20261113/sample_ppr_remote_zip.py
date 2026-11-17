"""Bounded public ZIP range retrieval of four preselected offline supervision triples."""
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile
from independent_coco import ROOT, digest

BASE = ROOT/'outputs/adaptive-reference-20261113'
OUT = BASE/'ppr-pixel-sample-v1'
ASSETS = {
    'source': ('1Bt_ishGRgIMSQVCCED4XfeTNI0KAQFU4', 12079675100, '.tif'),
    'target_a': ('1Yqtop2r4yyE09u9PCijYR4URwYkp8hof', 4639852620, '.tif'),
    'mask': ('1_cEuGXT_Yh8itZwxlvIeLM-oNkaG_CQr', 36915637, '.png'),
}


class RemoteZip(io.BufferedIOBase):
    def __init__(self, role, file_id, size, budget):
        self.role=role;self.file_id=file_id;self.size=size;self.budget=budget;self.position=0;self.etag=None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        assert 0<=self.position<=self.size
        return self.position
    def read(self, n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position)
        if not n:return b''
        assert n<=4*1024**2 and self.budget['bytes']+n<=32*1024**2
        assert len(self.budget['requests'])<96 and time.monotonic()-self.budget['start']<600
        start=self.position;end=start+n-1
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
        if self.etag:headers['If-Range']=self.etag
        url='https://drive.usercontent.google.com/download?id='+self.file_id+'&export=download&confirm=t'
        with self.opener.open(urllib.request.Request(url,headers=headers),timeout=40) as response:
            assert response.status==206, 'Server did not honor bounded range'
            assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
            etag=response.headers.get('ETag')
            if self.etag:assert etag==self.etag
            elif etag:self.etag=etag
            data=response.read(n+1);assert len(data)==n
        self.budget['bytes']+=n;self.position+=n
        self.budget['requests'].append(dict(role=self.role,start=start,end=end,bytes=n,sha256=hashlib.sha256(data).hexdigest(),etag=self.etag))
        return data


def main():
    OUT.mkdir(exist_ok=False)
    links_path=BASE/'ppr-group-exclusions-v3/results.json'
    assert digest(links_path)=='8783deef13634bd1433964d251e77af3cba765c280c5b4741cdf13318263f716'
    links=json.loads(links_path.read_text());rows=sorted((r for r in links['training_candidates'] if not r['exclude_for_known_consumed_group']),key=lambda r:tuple(map(int,r['ppr_candidate_id'].split('_'))))
    chosen=[];groups=set()
    for row in rows:
        if row['ppr_group'] in groups:continue
        chosen.append(row['ppr_candidate_id']);groups.add(row['ppr_group'])
        if len(chosen)==4:break
    assert len(chosen)==4
    protocol=Path(__file__).with_name('ppr-pixel-sample-protocol.md')
    plan=dict(ids=chosen,links_sha256=digest(links_path),driver_sha256=digest(__file__),protocol_sha256=digest(protocol),archives=ASSETS,budget_bytes=32*1024**2,scope='four known training-origin groups; offline source/target/mask qualification only')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (OUT/'protocol.md').write_bytes(protocol.read_bytes())
    report=dict(plan_sha256=digest(OUT/'plan.json'),files=[],requests=[],finished=False,pixels_decoded=False)
    budget=dict(start=time.monotonic(),bytes=0,requests=report['requests'])
    try:
        for role,(file_id,size,suffix) in ASSETS.items():
            remote=RemoteZip(role,file_id,size,budget)
            with zipfile.ZipFile(remote) as archive:
                infos=archive.infolist();index=[]
                for info in infos:
                    if info.is_dir():continue
                    index.append(dict(name=info.filename,bytes=info.file_size,compressed=info.compress_size,crc=info.CRC,offset=info.header_offset))
                (OUT/(role+'-index.json')).write_text(json.dumps(index,indent=2),encoding='utf-8')
                for key in chosen:
                    candidates=[i for i in infos if Path(i.filename).stem==key and Path(i.filename).suffix.lower() in ([suffix,'.tiff'] if suffix=='.tif' else [suffix])]
                    assert len(candidates)==1,(role,key,'Missing or ambiguous member')
                    member=candidates[0];assert member.file_size<=4*1024**2 and not member.flag_bits&1
                    raw=archive.read(member);assert len(raw)==member.file_size
                    folder=OUT/'samples'/role;folder.mkdir(parents=True,exist_ok=True)
                    path=folder/(key+suffix);path.write_bytes(raw)
                    report['files'].append(dict(role=role,id=key,member=member.filename,bytes=len(raw),crc=member.CRC,sha256=hashlib.sha256(raw).hexdigest(),local_path=str(path.relative_to(ROOT))))
            print(json.dumps(dict(role=role,files=len(report['files']),read_bytes=budget['bytes'])),flush=True)
        report['finished']=True
    finally:
        report.update(read_bytes=budget['bytes'],seconds=time.monotonic()-budget['start'])
        (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert len(report['files'])==12 and digest(__file__)==plan['driver_sha256']
    print(json.dumps(dict(finished=True,ids=chosen,read_bytes=budget['bytes'],requests=len(report['requests']))))


if __name__=='__main__':main()
