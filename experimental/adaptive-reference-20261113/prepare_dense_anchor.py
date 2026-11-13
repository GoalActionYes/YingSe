"""Fetch a fixed small feature model only, without remote Python execution."""
import hashlib
import json
from urllib.request import urlopen
from independent_coco import ROOT,fetch,digest

REVISION='ed25f3a31f01632728cabb09d1542f84ab7b0056'
WEIGHT_SHA='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
SMALL={'config.json':('5664b325e6258d3960fad8c4c1cff958f3cc2272',547),
       'preprocessor_config.json':('ff5b47c2edcd1d3556d63c01a65d93b58b9efce1',436),
       'README.md':('6b3380957df44ed203ec1d5102e1245accbbbba9',3033)}


def main():
    directory=ROOT/'.tools/dense-anchor-model'; directory.mkdir(exist_ok=True)
    files=[]
    for name,(oid,size) in SMALL.items():
        path=directory/name
        if path.is_file(): raw=path.read_bytes()
        else:
            with urlopen(f'https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/{name}',timeout=30) as response:
                raw=response.read(size+1)
        if len(raw)!=size or hashlib.sha1(f'blob {size}\0'.encode()+raw).hexdigest()!=oid:
            raise ValueError('Model asset Git blob mismatch')
        if not path.is_file(): path.write_bytes(raw)
        files.append(dict(path=name,sha256=digest(path),bytes=size))
    path=directory/'model.safetensors'
    fetch(f'https://huggingface.co/facebook/dinov2-small/resolve/{REVISION}/model.safetensors',
          path,WEIGHT_SHA,88249960,88249960)
    files.append(dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size))
    manifest=dict(repository='facebook/dinov2-small',revision=REVISION,files=files,
        license='Apache-2.0 per model-card metadata; retained model card is HF-authored, not original author card')
    (directory/'verified-assets.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__': main()
