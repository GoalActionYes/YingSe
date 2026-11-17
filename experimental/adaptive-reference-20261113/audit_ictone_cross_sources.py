"""Metadata-only TST/COCO provenance audit; no image downloads or inference."""
import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REV='63bf13833b75407c862d608c3d81281d0a35b61f'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1_048_576),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert Path(args.run).name==args.run and args.run not in ('.','..')
    out=ROOT/'outputs/adaptive-reference-20261113'/args.run
    out.mkdir(exist_ok=False)
    raw_source=json.loads((ROOT/'.downloads/tst100k-source.json').read_text())
    assert raw_source['metadata']['sha']==REV
    readme=ROOT/'.tools/TST100K-index/README.md'
    body=readme.read_bytes()
    expected=next(x['oid'] for x in raw_source['tree'] if x['path']=='README.md')
    assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==expected
    index=ROOT/'.tools/TST100K-index/triplet.json'
    assert digest(index)=='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413'
    triples=json.loads(index.read_text())
    content_paths={x['content'] for x in triples}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    files={}
    requests=[]
    total=0
    for folder in ['00','01']:
        url=f'https://huggingface.co/api/datasets/ToneStyle/TST100K/tree/{REV}/content_images/{folder}?recursive=false&limit=1000'
        page=0
        while url:
            page+=1
            assert len(requests)<24
            parsed=urllib.parse.urlparse(url)
            assert parsed.scheme=='https' and parsed.netloc=='huggingface.co'
            assert urllib.parse.unquote(parsed.path)==f'/api/datasets/ToneStyle/TST100K/tree/{REV}/content_images/{folder}'
            with opener.open(url,timeout=30) as response:
                data=response.read(2_000_001)
                link=response.headers.get('Link','')
            total+=len(data)
            assert len(data)<=2_000_000 and total<=12_000_000
            rows=json.loads(data)
            assert isinstance(rows,list)
            filename=f'{folder}-{page}.json'
            (out/filename).write_bytes(data)
            requests.append(dict(path=filename,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            for row in rows:
                assert row['type']=='file' and row['path'].startswith('content_images/'+folder+'/')
                assert row['path'] not in files and row['lfs']['size']==row['size']
                files[row['path']]=dict(sha256=row['lfs']['oid'],bytes=row['size'])
            match=re.search(r'<([^>]+)>;\s*rel="next"',link)
            url=match.group(1) if match else None
            print(json.dumps(dict(folder=folder,page=page,listed_files=len(files))),flush=True)
    assert content_paths<=set(files),sorted(content_paths-set(files))[:4]
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    manifest=json.loads((coco/'manifest.json').read_text())
    hashes={entry['sha256'] for entry in files.values()}
    matches=[]
    for image in manifest['images']:
        original=ROOT/'.downloads/coco/images'/image['file_name']
        assert digest(original)==image['original_sha256']
        assert digest(coco/(str(image['id'])+'-input.png'))==image['input_sha256']
        if image['original_sha256'] in hashes:matches.append(image['id'])
    report=dict(finished=True,scope='metadata and exact original-byte overlap only; no image/perceptual/training-access proof',
                revision=REV,index_sha256=digest(index),triplets=len(triples),used_content_paths=len(content_paths),
                complete_listed_content_files=len(files),all_used_content_paths_have_lfs_metadata=True,
                dataset_card_sha256=digest(readme),dataset_card_git_blob_verified=True,
                declared_tst100k_sources=['PPR10K','MIT-Adobe FiveK','Food-101'],
                metadata_requests=requests,metadata_download_bytes=total,
                coco_manifest_sha256=digest(coco/'manifest.json'),coco_images=len(manifest['images']),
                coco_pairs=len(manifest['pairs']),coco_exact_original_byte_matches=matches,
                coco_all30_already_consumed=True,near_duplicate_or_reencoded_overlap_not_excluded=True,
                backbone_pretraining_overlap_unknown=True,checkpoint_exact_training_rows_unknown=True,
                source_sha256=digest(Path(__file__)))
    (out/'content-file-hashes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    report['content_hash_list_sha256']=digest(out/'content-file-hashes.json')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='metadata_requests'},indent=2))


if __name__=='__main__':
    main()
