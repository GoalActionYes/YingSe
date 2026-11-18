"""Select and fetch the fixed small multi-room learning cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from audit_synclight_real_intake import REVISION, archive_index
from fetch_synclight_real_pixels import fetch


def run(source, out, proxy, batch='original'):
    manifest_path = source/'manifest.jsonl'
    manifest = [json.loads(line) for line in manifest_path.read_bytes().splitlines()]
    train = json.loads((source/'train.json').read_bytes())
    meta = json.loads((source/'meta.json').read_bytes())
    assert meta['sha']==REVISION and meta['cardData']['license']=='cc-by-4.0' and not meta['gated']
    tree = {row['path']:row for row in train}
    pool = sorted((r for r in manifest if r['source']=='real' and r['split']=='train' and r['n_raw']>=4
                   and r['scene'] not in ('scene35','scene25')),key=lambda row:(row['bytes'],row['scene']))
    assert [r['scene'] for r in pool[:10]]==['scene22','scene18','scene37','scene24','scene23','scene36',
                                             'scene2','scene21','scene1','scene12']
    selection = {'original':pool[:4], 'extension':[pool[1],*pool[4:6]], 'next':pool[6:10]}[batch]
    for row in selection:
        released=tree[row['path']]
        assert released['size']==row['bytes'] and released['lfs']['oid']==row['sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:
        indexed=list(pool.map(lambda row:archive_index(row,proxy),selection))
    tasks=[]
    omitted=[]
    for row,index in zip(selection,indexed,strict=True):
        scene=row['scene']
        required = {f'{scene}_1_{role}' for role in ('0_ambient','1_camvis')}
        formats = ('.DNG',) if batch=='original' else ('.DNG','.CR2')
        matches = [m for m in index['file_members']
                   if Path(m['path']).stem in required and Path(m['path']).suffix.upper() in formats]
        matched_paths = [Path(m['path']) for m in matches]
        if len(matches)==2 and {p.stem for p in matched_paths}==required and len({p.suffix for p in matched_paths})==1:
            tasks.extend((index,member) for member in matches)
        else:
            omitted.append(scene)
    budget=sum(member['size'] for _,member in tasks)
    header_bytes=sum(req['size'] for row in indexed for req in row['requests'])
    assert 0<budget<=160*1024**2 and header_bytes<=1024**2
    out.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        files=dict(pool.map(lambda task:fetch(*task,out,proxy),tasks))
    record=dict(revision=REVISION,selected=[row['scene'] for row in selection],
                omitted=omitted,bytes=budget,header_bytes=header_bytes,files=files,
                index=dict(zip((row['scene'] for row in selection),indexed,strict=True)),
                manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest())
    if batch=='next':
        record['batch']=batch
    (out/'intake.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(list(files),budget)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--proxy')
    batch=parser.add_mutually_exclusive_group()
    batch.add_argument('--extend',action='store_true')
    batch.add_argument('--next',action='store_true')
    args=parser.parse_args()
    run(args.source,args.out,args.proxy,'next' if args.next else 'extension' if args.extend else 'original')
