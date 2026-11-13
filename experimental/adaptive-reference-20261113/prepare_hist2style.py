"""Fetch only pinned research inference assets; no dataset or API use."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from independent_coco import ROOT,digest


def main():
    meta=json.loads((ROOT/'.downloads/hist2style-source.json').read_text(encoding='utf-8'))
    revision='a3ef9ba0455d957b54c459c88cac36cf5740c164'
    if meta['revision']!=revision:raise ValueError('Unexpected source revision')
    names=['LICENSE','README.md','run.py','demo/model.py','demo/utils.py','demo/model-epoch=1500-val_loss=2.79.ckpt']
    out=ROOT/'.tools/Hist2Style-source';out.mkdir(exist_ok=True);records=[]
    for name in names:
        entry=next(x for x in meta['tree']['tree'] if x['path']==name)
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raw=path.read_bytes()
        else:
            with urlopen(f'https://raw.githubusercontent.com/dgalor/hist2style/{revision}/{name}',timeout=60) as response:raw=response.read(8*1024*1024)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if len(raw)!=entry['size'] or blob!=entry['sha']:raise ValueError('Git blob mismatch: '+name)
        if not path.exists():path.write_bytes(raw)
        records.append(dict(path=name,bytes=len(raw),sha256=digest(path),git_blob=blob))
    (out/'manifest.json').write_text(json.dumps(dict(repository='https://github.com/dgalor/hist2style',revision=revision,
        license='Adobe Research License, non-commercial research only; not product dependency',files=records),indent=2),encoding='utf-8')
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
