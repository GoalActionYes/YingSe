"""Download only reviewed pinned model files; preserve partials and verify identity."""
import argparse,json,hashlib,time
from pathlib import Path
import requests
from independent_coco import ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proxy');args=parser.parse_args()
    root=ROOT/'.tools/StyleGallery-audit';manifest=json.loads((ROOT/'docs/research/2026-11-17-stylegallery-assets.json').read_text())
    session=requests.Session()
    if args.proxy:session.proxies.update({'http':args.proxy,'https':args.proxy})
    results=[]
    for record in manifest['models']:
        folder=root/record['target'];folder.mkdir(parents=True,exist_ok=True)
        for entry in record['files']:
            name=entry['rfilename'];path=folder/name;assert path.resolve().is_relative_to(folder.resolve());path.parent.mkdir(parents=True,exist_ok=True)
            sha=entry.get('lfs',{}).get('sha256');blob=entry.get('blobId');assert sha or blob,name
            def verify(p):
                assert p.stat().st_size==entry['size'],name
                h=hashlib.sha256() if sha else hashlib.sha1()
                if not sha:h.update(('blob '+str(entry['size'])+'\0').encode())
                with p.open('rb') as source:
                    for chunk in iter(lambda:source.read(1024*1024),b''):h.update(chunk)
                assert h.hexdigest()==(sha or blob),name
                return h.hexdigest()
            if not path.exists():
                part=path.with_name(path.name+'.part');position=part.stat().st_size if part.exists() else 0;assert position<=entry['size']
                if position<entry['size']:
                    url='https://huggingface.co/'+record['repository']+'/resolve/'+record['revision']+'/'+name
                    headers={'Range':f'bytes={position}-'} if position else {}
                    with session.get(url,headers=headers,stream=True,timeout=(30,60)) as response:
                        response.raise_for_status()
                        if position:assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {position}-'),name
                        with part.open('ab' if position else 'xb') as target:
                            tick=time.monotonic()
                            for chunk in response.iter_content(1024*1024):
                                if not chunk:continue
                                target.write(chunk);position+=len(chunk);assert position<=entry['size'],name
                                if time.monotonic()-tick>=20:print(json.dumps(dict(file=name,bytes=position,total=entry['size'])),flush=True);tick=time.monotonic()
                verify(part);part.rename(path)
            identity=verify(path);results.append(dict(repository=record['repository'],path=name,size=entry['size'],identity=identity));print('verified',record['repository'],name,entry['size'],flush=True)
        (root/'download-progress.json').write_text(json.dumps(dict(files=results,finished=False),indent=2))
    (root/'download-results.json').write_text(json.dumps(dict(files=results,finished=True),indent=2));print('All pinned assets verified',flush=True)


if __name__=='__main__':main()
