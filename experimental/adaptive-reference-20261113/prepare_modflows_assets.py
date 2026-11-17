"""Recreate only pinned public assets; verify existing files without overwriting."""
import hashlib,json,zipfile
from pathlib import Path
import requests
from independent_coco import ROOT,digest


def main():
    evidence=json.loads((ROOT/'docs/research/2026-11-17-modflows-evidence.json').read_text())
    root=ROOT/'.tools/ModFlows-audit';root.mkdir(exist_ok=True);session=requests.Session()
    def obtain(url,path,sha,size=None):
        if path.exists():
            assert digest(path)==sha and (size is None or path.stat().st_size==size),str(path)
            return
        path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part');assert not part.exists(),str(part)
        h=hashlib.sha256();total=0
        with session.get(url,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            with part.open('xb') as target:
                for chunk in response.iter_content(1024*1024):
                    if chunk:target.write(chunk);h.update(chunk);total+=len(chunk)
        assert h.hexdigest()==sha and (size is None or total==size),str(part)
        part.rename(path)
    source=evidence['author_source']
    for e in source['files']:
        path=root/'source'/e['path'];assert path.resolve().is_relative_to((root/'source').resolve())
        obtain('https://raw.githubusercontent.com/maria-larchenko/modflows/'+source['commit']+'/'+e['path'],path,e['sha256'],e['size'])
        data=path.read_bytes();assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()==e['git_blob_sha1']
    weight=evidence['weight'];obtain('https://huggingface.co/'+weight['repository']+'/resolve/'+weight['revision']+'/'+weight['name'],root/weight['name'],weight['sha256'],weight['size'])
    vendor=root/'vendor';vendor.mkdir(exist_ok=True)
    for e in evidence['dependencies']:
        wheel=root/'wheels'/e['filename'];obtain(e['url'],wheel,e['sha256'],e['size'])
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                path=vendor/name;assert path.resolve().is_relative_to(vendor.resolve())
                if name.endswith('/'):continue
                data=archive.read(name)
                if path.exists():assert path.read_bytes()==data,str(path)
                else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name,value in [('source-manifest.json',source),('weight-manifest.json',weight),('dependency-plan.json',evidence['dependencies'])]:
        path=root/name
        if path.exists():assert json.loads(path.read_text())==value
        else:path.write_text(json.dumps(value,indent=2))
    print('Verified pinned source, checkpoint and all isolated dependency files; existing assets preserved')


if __name__=='__main__':main()
