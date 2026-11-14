"""Fit one median radius from initial training content/reference inputs only."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from radius_context import calibrated_radius


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c' or digest(checkpoint)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Changed calibration inputs')
    lookup={v['original_path']:v for v in prep['files']};rows=cohort['groups']['train'];out=base/'context-radius-calibration-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['calibrate_context_radius.py','radius_context.py','test_radius_context.py','radius-context-protocol.md','train_triplet_pilot.py','independent_coco.py','appearance_pairwise.py','pairwise_context.py','reference_context.py','training_checks.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Initial current13 training content/reference input-only radius calibration, no GT or dev/reserved pixels',source_sha256={p.name:digest(p) for p in frozen.iterdir()},cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoint_sha256=digest(checkpoint),inputs=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');model,api,_=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    with torch.no_grad():
        for offset in range(0,len(rows),4):
            images=[];items=[]
            for row in rows[offset:offset+4]:
                for role in ['content','reference']:
                    entry=lookup[row[role]];p=data/entry['prepared_path']
                    if digest(p)!=entry['sha256']:raise ValueError('Training input changed')
                    with Image.open(p) as im:images.append(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
                    items.append(dict(index=row['index'],role=role,path=row[role],sha256=entry['sha256']))
            rgb=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().cuda()/255;norms=model.img_projection(api['TF'].resize(rgb,(64,64),antialias=True)).mean((-2,-1)).norm(dim=1).cpu().numpy()
            for item,value in zip(items,norms):report['inputs'].append(dict(item,norm=float(value)))
            if (offset+4)%256==0:print(json.dumps(dict(train_rows_calibrated=offset+4)),flush=True)
    values=[v['norm'] for v in report['inputs']];radius=calibrated_radius(values)
    if len(values)!=2048 or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(checkpoint)!=report['checkpoint_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()) or any(digest(data/lookup[v['path']]['prepared_path'])!=v['sha256'] for v in report['inputs']):raise ValueError('Calibration scope/evidence changed')
    report.update(radius=radius,minimum_norm=float(min(values)),maximum_norm=float(max(values)),finished=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,radius=radius,minimum_norm=min(values),maximum_norm=max(values))),flush=True)


if __name__=='__main__':main()
