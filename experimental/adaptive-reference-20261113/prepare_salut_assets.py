"""Fetch pinned author research assets; no training, installer, or executable is run."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path,PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen
import zipfile
from independent_coco import ROOT,fetch,digest

SOURCE='3e62f9c9b59ea7b30d220ff6ba61047dab898c9a'
SOURCE_SHA='3ae78661d89522cc73215719a4d412d7a8dc7ca5065863ccff322d907e120963'
MODEL='940cab98d7a37bea828cfb6efa2433ece5bb423a'
DATA='6ed86e34c37c558a0b2401d52f06603d619472e1'
WEIGHT='epoch=100-step=4127466.ckpt.state.pt'
WEIGHT_SHA='5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b'


def main():
    cache=ROOT/'.downloads/salut';cache.mkdir(parents=True,exist_ok=True)
    archive=cache/'source.zip'
    fetch(f'https://codeload.github.com/Ry3nG/SA-LUT/zip/{SOURCE}',archive,SOURCE_SHA,maximum=200*1024*1024)
    destination=(ROOT/'.tools/SA-LUT-source').resolve();destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            parts=PurePosixPath(item.filename).parts
            if not parts or parts[0]!='SA-LUT-'+SOURCE or '..' in parts or '\\' in item.filename:raise ValueError('Unexpected archive path')
            if len(parts)==1:continue
            target=destination.joinpath(*parts[1:]).resolve()
            if not target.is_relative_to(destination):raise ValueError('Archive escapes author source directory')
            if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Source archive symlink is not extracted')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as stream:target.write_bytes(stream.read())
    fetch(f'https://huggingface.co/zrgong/SA-LUT/resolve/{MODEL}/{quote(WEIGHT)}',
          cache/WEIGHT,WEIGHT_SHA,218400931,218400931)
    out=ROOT/'outputs/adaptive-reference-20261113/pst50';out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for folder,prefix in [('content_709','in'),('paired_style','tar'),('paired_gt','gt')]:
        with urlopen(f'https://huggingface.co/api/datasets/zrgong/PST50/tree/{DATA}/{folder}',timeout=30) as response:listing=json.load(response)
        entries.extend(x for x in listing if x['path'] in {f'{folder}/{prefix}{i}.png' for i in range(1,6)})
    if len(entries)!=15:raise ValueError('Expected exact first-five paired assets')
    def prepare(entry):
        path=out/entry['path'];sha=entry['lfs']['oid'];size=entry['size']
        fetch(f"https://huggingface.co/datasets/zrgong/PST50/resolve/{DATA}/{quote(entry['path'])}",path,sha,size,30*1024*1024)
        return dict(path=entry['path'],sha256=sha,bytes=size)
    with ThreadPoolExecutor(max_workers=3) as pool:images=list(pool.map(prepare,entries))
    report=dict(source_commit=SOURCE,source_archive_sha256=digest(archive),model_revision=MODEL,
                checkpoint_sha256=WEIGHT_SHA,dataset_revision=DATA,development_pairs=list(range(1,6)),
                untouched_evaluation_pairs=list(range(6,51)),images=images,
                source_license='S-Lab License 1.0 non-commercial; retained with source, not commercial product dependency',
                dataset_license_metadata='CC-BY-4.0 as listed by author dataset card',
                scope='Assets prepared only; author inference and Windows interpolation compatibility not yet verified')
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2))


if __name__=='__main__':main()
