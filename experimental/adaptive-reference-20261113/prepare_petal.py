"""Acquire bounded official inference assets; no installers or global env changes."""
import hashlib,json,sys
from pathlib import Path
from urllib.request import Request,urlopen
from independent_coco import ROOT,digest,fetch


def metadata(url):
    with urlopen(Request(url,headers={'User-Agent':'YingSe-academic-research'}),timeout=30) as response:raw=response.read(2*1024*1024+1)
    if len(raw)>2*1024*1024:raise ValueError('Metadata exceeds bound')
    return json.loads(raw)


def main():
    out=ROOT/'.tools/PETAL-source';out.mkdir(parents=True,exist_ok=True)
    commit='162bad662d5a108285bb308699faed2888e3d782'
    previous=out/'manifest.json'
    if previous.is_file() and json.loads(previous.read_text())['revision']!=commit:raise ValueError('Do not update an existing source snapshot implicitly')
    tree=metadata(f'https://api.github.com/repos/jiacongfang/PETAL/git/trees/{commit}?recursive=1')
    if tree.get('truncated'):raise ValueError('Incomplete source tree')
    names={'infer.py','models_lut.py','models_style_encoder.py','README.md','LICENSE','LICENSE-DINOv3.md'}
    files=[]
    for entry in tree['tree']:
        name=entry['path']
        if entry['type']!='blob' or not (name in names or (name.startswith('layers/') and name.endswith('.py'))):continue
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or entry['size']>1024*1024:raise ValueError('Unsafe source member')
        path=out/relative;url=f'https://raw.githubusercontent.com/jiacongfang/PETAL/{commit}/{name}'
        fetch(url,path,size=entry['size'],maximum=1024*1024)
        raw=path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['sha']:raise ValueError('Git blob mismatch')
        files.append(dict(path=name,url=url,bytes=len(raw),git_blob=entry['sha'],sha256=digest(path)))
    if not names.issubset({r['path'] for r in files}):raise ValueError('Missing inference source/license')
    model_revision='34eb5176def5f583b72306fea03e0f6b44c6f3bf'
    weights=[dict(path='ckpts/style_encoder.pt',bytes=1449783062,sha256='c50e00cca2e1dae4cc9a02f6217b0c2f518eb438ab7356beea3d050c7d6d8852'),dict(path='ckpts/lut_model.pt',bytes=6921051,sha256='b4ba53ed85139b983f59cd364d25bf8d3e62867ee259d919c985773be3061c13')]
    manifest=dict(repo='https://github.com/jiacongfang/PETAL',revision=commit,model_repo='JiacongFang/PETAL',model_revision=model_revision,files=files,weights=weights,ready=False,scope='Official source/weights acquired for isolated noncommercial academic inference; DINOv3-derived license retained, not product integration')
    previous.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    # Pure-Python wheel is imported from this isolated file, not installed.
    package=metadata('https://pypi.org/pypi/kornia/0.7.0/json')
    wheels=[v for v in package['urls'] if v['filename']=='kornia-0.7.0-py2.py3-none-any.whl']
    if len(wheels)!=1:raise ValueError('Unexpected Kornia wheel inventory')
    wheel=wheels[0];dep=out/'deps'/wheel['filename']
    fetch(wheel['url'],dep,sha=wheel['digests']['sha256'],size=wheel['size'],maximum=4*1024*1024)
    manifest['kornia']=dict(version='0.7.0',path=str(dep.relative_to(out)),sha256=digest(dep),url=wheel['url'],installed=False)
    previous.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for row in weights:
        url=f'https://huggingface.co/JiacongFang/PETAL/resolve/{model_revision}/{row["path"]}'
        fetch(url,out/row['path'],sha=row['sha256'],size=row['bytes'],maximum=1500*1024*1024)
        row['url']=url;print(json.dumps(dict(asset=row['path'],bytes=row['bytes'],verified=True)),flush=True)
    manifest['ready']=True;previous.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(ready=True,revision=commit,source_files=len(files),weights=len(weights),kornia_installed=False)),flush=True)


if __name__=='__main__':main()
