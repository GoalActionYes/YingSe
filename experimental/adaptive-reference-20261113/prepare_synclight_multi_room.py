"""Select and fetch the fixed small multi-room learning cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from audit_synclight_real_intake import REVISION, archive_index
from fetch_synclight_real_pixels import fetch


def run(source, out, proxy):
    manifest_path = source/'manifest.jsonl'
    manifest = [json.loads(line) for line in manifest_path.read_bytes().splitlines()]
    train = json.loads((source/'train.json').read_bytes())
    meta = json.loads((source/'meta.json').read_bytes())
    assert meta['sha']==REVISION and meta['cardData']['license']=='cc-by-4.0' and not meta['gated']
    tree = {row['path']:row for row in train}
    selection = sorted((r for r in manifest if r['source']=='real' and r['split']=='train' and r['n_raw']>=4
                        and r['scene'] not in ('scene35','scene25')),key=lambda row:(row['bytes'],row['scene']))[:4]
    assert len(selection)==4
    for row in selection:
        released=tree[row['path']]
        assert released['size']==row['bytes'] and released['lfs']['oid']==row['sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:
        indexed=list(pool.map(lambda row:archive_index(row,proxy),selection))
    tasks=[]
    omitted=[]
    for row,index in zip(selection,indexed,strict=True):
        scene=row['scene']
        required={f'{scene}_1_{role}.DNG' for role in ('0_ambient','1_camvis')}
        matches=[member for member in index['file_members'] if Path(member['path']).name in required]
        if len(matches)==2 and {Path(m['path']).name for m in matches}==required:
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
    (out/'intake.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(list(files),budget)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--proxy')
    args=parser.parse_args()
    run(args.source,args.out,args.proxy)
