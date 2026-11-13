"""Re-evaluate raw finite outputs and saved validation pixels for every epoch."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v1';run=base/'training-v1'
    result=json.loads((run/'results.json').read_text(encoding='utf-8'));cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'));lookup={r['original_path']:data/r['prepared_path'] for r in manifest['files']}
    if not result['finished'] or result['epochs'][-1]['updates']!=256:raise ValueError('Training incomplete')
    for name,sha in result['source_sha256'].items():
        if digest(run/'frozen-source'/name)!=sha:raise ValueError('Frozen training source changed')
    inputs=[]
    for part in ['train','validation']:
        for row in cohort['groups'][part]:
            tensors=[]
            for role in ['content','reference']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                tensors.append(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/255)
            inputs.append((part,row['index'],*tensors))
    model,api,_=author_model();model.eval();epochs=[]
    with torch.inference_mode():
        for epoch in range(9):
            checkpoint=run/f'epoch-{epoch}.pth';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
            model.load_state_dict(state,strict=True);maxdiff=0;low=float('inf');high=-float('inf')
            for part,index,c,s in inputs:
                hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
                raw=finite_output(model(c,hc,hs)[0],c.shape);low=min(low,float(raw.min()));high=max(high,float(raw.max()))
                if part=='validation':
                    pixels=np.rint(raw[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
                    folder='before' if epoch==0 else f'epoch-{epoch}'
                    with Image.open(run/folder/f'{index}.png') as im:original=np.array(im)
                    maxdiff=max(maxdiff,int(np.abs(pixels.astype(int)-original.astype(int)).max()))
            epochs.append(dict(epoch=epoch,checkpoint_sha256=digest(checkpoint),raw_min=low,raw_max=high,finite_cases=len(inputs),saved_validation_max_difference=maxdiff))
            print(json.dumps(epochs[-1]),flush=True)
    best=result['best_epoch'];scores=json.loads((run/f'epoch-{best}/scores.json').read_text(encoding='utf-8'));current={r['index']:r['mae'] for r in result['current']['cases']}
    baseline=np.array([current[r['index']] for r in scores['cases']]);actual=np.array([r['mae'] for r in scores['cases']])
    summary=dict(count=32,current_mean=float(baseline.mean()),candidate_mean=float(actual.mean()),improved=int((actual<baseline-.1).sum()),regressed=int((actual>baseline+.1).sum()),severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum()))
    report=dict(scope='All 9 checkpoints raw-finite audit on 128 train +32 development validation, no reserved evaluation',
        source_sha256=digest(Path(__file__)),training_results_sha256=digest(run/'results.json'),epochs=epochs,summary=summary,
        passed=all(r['saved_validation_max_difference']==0 for r in epochs),product_integrated=False)
    (run/'finite-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
