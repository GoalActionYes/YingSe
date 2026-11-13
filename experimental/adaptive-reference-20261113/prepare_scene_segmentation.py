"""Pinned SegFormer research assets, safe tensors only."""
import json
from urllib.request import urlopen
from independent_coco import ROOT,digest,fetch


def main():
    meta=json.loads((ROOT/'.downloads/segformer-source.json').read_text(encoding='utf-8'))
    revision='489d5cd81a0b59fab9b7ea758d3548ebe99677da'
    if meta['metadata']['sha']!=revision:raise ValueError('Unexpected revision')
    out=ROOT/'.tools/scene-segformer';out.mkdir(exist_ok=True);files=[]
    for name in ['README.md','config.json','preprocessor_config.json','model.safetensors']:
        entry=next(x for x in meta['tree'] if x['path']==name)
        fetch(f'https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/{revision}/{name}',out/name,
            entry.get('lfs',{}).get('oid'),entry['size'])
        files.append(dict(path=name,sha256=digest(out/name),bytes=entry['size']))
    # Retain the original upstream license; academic experiment only.
    with urlopen('https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE',timeout=30) as response:license_bytes=response.read(30000)
    (out/'UPSTREAM-LICENSE').write_bytes(license_bytes)
    files.append(dict(path='UPSTREAM-LICENSE',sha256=digest(out/'UPSTREAM-LICENSE'),bytes=len(license_bytes)))
    (out/'manifest.json').write_text(json.dumps(dict(repository='nvidia/segformer-b0-finetuned-ade-512-512',revision=revision,
        files=files,scope='Scene-label evidence for noncommercial research, not a grading model or product dependency'),indent=2),encoding='utf-8')
    print(json.dumps(files),flush=True)


if __name__=='__main__':main()
