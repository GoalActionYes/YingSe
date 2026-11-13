"""Frozen continuation experiment on existing development data, never reserved."""
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output

START_SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def main():
    config=dict(seed=20261114,lr=1e-5,weight_decay=1e-4,batch=4,extra_epochs=24,max_updates=768,
        start_epoch=8,evaluate_epochs=[12,16,20,24,28,32],gradient_weight=.1,gradient_clip=1.,optimizer_state='reset AdamW')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v1';checkpoint=base/'training-v1/epoch-8.pth'
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed starting checkpoint')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));prep_path=data/'srgb-v1/manifest.json';prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(data/'cohort.json'):raise ValueError('Invalid prepared cohort')
    lookup={entry['original_path']:data/entry['prepared_path'] for entry in prep['files']}
    for entry in prep['files']:
        if digest(lookup[entry['original_path']])!=entry['sha256']:raise ValueError('Changed prepared file')
    out=base/'training-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['train_triplet_extension.py','triplet-extension-protocol.md','train_triplet_pilot.py','training_checks.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(prep_path),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},scope='Development only; no reserved data, no app replacement',epochs=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,author=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite start weights')
    model.load_state_dict(state,strict=True);samples={}
    for part in ['train','validation']:
        samples[part]=[]
        for row in cohort['groups'][part]:
            tensors=[]
            for role in ['content','reference','gt']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                tensors.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
            samples[part].append((row,*tensors))
    if len(samples['train'])!=128 or len(samples['validation'])!=32:raise ValueError('Unexpected sample count')
    def forward(c,s):
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        return finite_output(model(c,hc,hs)[0],c.shape).clamp(0,1)
    def pixels(value):return np.rint(finite_output(value,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();folder=out/f'epoch-{epoch}';folder.mkdir();rows=[]
        for row,c,s,gt in samples['validation']:
            prediction=pixels(forward(c[None].cuda(),s[None].cuda())[0]);path=folder/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            rows.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-pixels(gt)).mean()),sha256=digest(path)))
        result=dict(mean=float(np.mean([r['mae'] for r in rows])),cases=rows)
        (folder/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate(8);best=baseline['mean'];best_epoch=8;report.update(baseline=baseline,author=author)
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'baseline':best}),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);rng=np.random.default_rng(config['seed']);updates=0
    for epoch in range(9,33):
        model.train();start=time.perf_counter();losses=[];order=rng.permutation(len(samples['train']))
        for offset in range(0,len(order),4):
            chosen=[samples['train'][int(j)] for j in order[offset:offset+4]]
            c,s,gt=[torch.stack([r[k] for r in chosen]).cuda() for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);pred=forward(c,s)
            loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path)
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,checkpoint_sha256=digest(path))
        if epoch in config['evaluate_epochs']:
            validation=evaluate(epoch);record['validation_mean']=validation['mean']
            if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best,finished=epoch==32)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=config['max_updates']:raise ValueError('Unexpected update count')
    selected=json.loads((out/f'epoch-{best_epoch}/scores.json').read_text(encoding='utf-8'))
    previous={r['index']:r['mae'] for r in baseline['cases']};improved=sum(previous[r['index']]-r['mae']>.1 for r in selected['cases'])
    severe=sum(r['mae']-previous[r['index']]>5 and r['mae']>previous[r['index']]*1.25 for r in selected['cases'])
    gate=dict(relative_improvement=1-best/baseline['mean'],improved=improved,severe_regressions=severe)
    gate['proceed_to_independent_evaluation']=gate['relative_improvement']>=.05 and improved>=20 and severe<=1
    report['development_gate']=gate;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
