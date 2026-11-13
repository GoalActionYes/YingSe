"""Audit all frozen grouped GT changes and fixed-model development response."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from multiref_response import response_metrics
from train_triplet_pilot import author_model
from training_checks import finite_output


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1'
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path):raise ValueError('Unverified prepared data')
    lookup={r['original_path']:data/r['prepared_path'] for r in prep['files']}
    for r in prep['files']:
        if digest(lookup[r['original_path']])!=r['sha256']:raise ValueError('Prepared bytes changed')
    checkpoints={
        'app_epoch8':('training-v1/epoch-8.pth','d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'),
        'expanded_epoch16':('training-expanded-control-v1/epoch-16.pth','17be34dfff3e778761c92e1442e75e8ce3bc2b0bb033543171eae40bc8e0b795')}
    for relative,sha in checkpoints.values():
        if digest(base/relative)!=sha:raise ValueError('Changed fixed model')
    out=base/'multiref-signal-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_multiref_signal.py','multiref_response.py','train_triplet_pilot.py','training_checks.py','multiref-training-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoints=checkpoints,
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},scope='Development signal only; no held-out evaluation, training or product promotion',finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(lookup[path]) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    report['gt_variation']={};validation=None
    for part,count,styles in [('train',256,4),('validation',32,2)]:
        groups=defaultdict(list)
        for row in cohort['groups'][part]:groups[row['group_id']].append(row)
        if len(groups)!=count or any(sorted(r['style_index'] for r in rows)!=list(range(styles)) for rows in groups.values()):raise ValueError('Incomplete grouped stream')
        variations=[]
        for group,rows in groups.items():
            rows=sorted(rows,key=lambda r:r['style_index']);a=image(rows[0]['gt'])
            for row in rows[1:]:
                b=image(row['gt']);metrics=response_metrics(a,a,a,b)
                variations.append(dict(group_id=group,style_index=row['style_index'],gt_change_mae=metrics['gt_change_mae']))
        values=[r['gt_change_mae'] for r in variations]
        report['gt_variation'][part]=dict(groups=count,comparisons=len(values),mean=float(np.mean(values)),minimum=float(min(values)),
            zero_change=sum(v==0 for v in values),below_one_code=sum(v<1 for v in values),cases=variations)
        if part=='validation':validation=groups
    model,api,_=author_model();report['models']={}
    for name,(relative,sha) in checkpoints.items():
        state=torch.load(base/relative,map_location='cpu',weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite model')
        model.load_state_dict(state,strict=True);model.eval();directory=out/name;directory.mkdir();cases=[];paired=[]
        with torch.inference_mode():
            for group,rows in validation.items():
                predictions=[];gts=[]
                for row in sorted(rows,key=lambda r:r['style_index']):
                    c,s=[(torch.from_numpy(image(row[role]).copy()).permute(2,0,1).float()[None]/255).cuda() for role in ['content','reference']]
                    hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
                    raw,grid=model(c,hc,hs);finite_output(grid,(1,13,8,16,16));finite_output(raw,c.shape)
                    pixels=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8);gt=image(row['gt'])
                    path=directory/f'{row["index"]}.png';Image.fromarray(pixels).save(path)
                    cases.append(dict(index=row['index'],group_id=group,style_index=row['style_index'],mae=float(np.abs(pixels.astype(float)-gt).mean()),sha256=digest(path)))
                    predictions.append(pixels);gts.append(gt)
                paired.append(dict(group_id=group,**response_metrics(*predictions,*gts)))
        means={key:float(np.mean([r[key] for r in paired])) for key in ['gt_change_mae','prediction_change_mae','delta_mae']}
        report['models'][name]=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,paired_means=means,
            original32_mean=float(np.mean([r['mae'] for r in cases if r['style_index']==0])),alternate32_mean=float(np.mean([r['mae'] for r in cases if r['style_index']==1])))
        print(json.dumps(dict(model=name,mean=report['models'][name]['mean'],paired_means=means)),flush=True)
    report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,variation={k:{n:v for n,v in r.items() if n!='cases'} for k,r in report['gt_variation'].items()})),flush=True)


if __name__=='__main__':main()
