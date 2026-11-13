"""Small, fixed academic-data audit; no heldout benchmark access or training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest,fetch


def main():
    revision='63bf13833b75407c862d608c3d81281d0a35b61f'
    metadata=json.loads((ROOT/'.downloads/tst100k-source.json').read_text(encoding='utf-8'))
    if metadata['metadata']['sha']!=revision:raise ValueError('Unexpected dataset revision')
    index=ROOT/'.tools/TST100K-index/triplet.json'
    entry=next(x for x in metadata['tree'] if x['path']=='triplet.json')
    if index.stat().st_size!=entry['size']:raise ValueError('Index size mismatch')
    if entry.get('lfs') and digest(index)!=entry['lfs']['oid']:raise ValueError('Index hash mismatch')
    records=json.loads(index.read_text(encoding='utf-8'))
    selected=records[:4];files=json.loads((ROOT/'.downloads/tst100k-audit-files.json').read_text())
    wanted={v for row in selected for v in row.values()}
    if {x['path'] for x in files}!=wanted:raise ValueError('Audit file list mismatch')
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/audit-v1';out.mkdir(parents=True,exist_ok=False)
    inputs=out/'inputs'
    def prepare(item):
        path=inputs/item['path']
        if not path.resolve().is_relative_to(inputs.resolve()):raise ValueError('Unsafe dataset path')
        expected=item.get('lfs',{}).get('oid')
        fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{revision}/{quote(item["path"])}',path,expected,item['size'])
        if not expected:
            raw=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if blob!=item['oid']:raise ValueError('Git asset hash mismatch')
        return dict(path=item['path'],sha256=digest(path),bytes=path.stat().st_size)
    with ThreadPoolExecutor(max_workers=3) as pool:assets=list(pool.map(prepare,files))
    rows=[]
    for i,row in enumerate(selected):
        decoded={};modes={};sizes={}
        for role,relative in row.items():
            with Image.open(inputs/relative) as image:
                modes[role]=image.mode;sizes[role]=list(image.size);decoded[role]=np.array(image.convert('RGB'))
        content,gt=decoded['content'],decoded['gt'];same_shape=content.shape==gt.shape
        diagnostics=dict(index=i,triplet=row,modes=modes,sizes=sizes,same_content_gt_shape=same_shape)
        if same_shape:
            # Oracle fit uses GT only to assess this data, never as inference.
            x=np.c_[content.reshape(-1,3)[::17].astype(float)/255,np.ones(len(content.reshape(-1,3)[::17]))]
            y=gt.reshape(-1,3)[::17].astype(float)/255
            matrix=np.linalg.lstsq(x,y,rcond=None)[0]
            predicted=np.clip(content.astype(float)/255@matrix[:3]+matrix[3],0,1)
            diagnostics.update(affine_oracle_mae=float(np.abs(predicted*255-gt).mean()),input_to_gt_mae=float(np.abs(content.astype(float)-gt).mean()))
        panel=Image.new('RGB',(1080,300),'#222222');draw=ImageDraw.Draw(panel)
        for n,role in enumerate(['content','reference','gt']):
            image=Image.fromarray(decoded[role]);image.thumbnail((356,260));panel.paste(image,(n*360,25));draw.text((n*360+8,6),role,fill='white')
        panel.save(out/f'{i}-triplet.jpg',quality=95);rows.append(diagnostics)
    report=dict(dataset='ToneStyle/TST100K',revision=revision,index_sha256=digest(index),total_triplets=len(records),selected_indices=list(range(4)),
        scope='Four training-data audit samples only; not independent evaluation, not model training or product validation',
        source_sha256=digest(Path(__file__)),license='Non-commercial academic/research only; excludes commercial product development',assets=assets,cases=rows)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
