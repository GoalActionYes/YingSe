"""Matched continuation with an optional zero-initialized appearance context."""
import argparse
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
from training_gates import development_gate
from reference_context import enable_reference_context,reference_forward

START_SHA='17be34dfff3e778761c92e1442e75e8ce3bc2b0bb033543171eae40bc8e0b795'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--appearance',action='store_true');args=parser.parse_args()
    config=dict(seed=20261116,base_lr=1e-5,new_head_lr=1e-4,weight_decay=1e-4,batch=4,start_epoch=16,epochs=8,max_updates=1024,gradient_weight=.1,gradient_clip=1.)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v2';anchor=base/'training-expanded-control-v1'
    checkpoint=anchor/'epoch-16.pth'
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed starting weights')
    anchor_path=anchor/'results.json';anchor_report=json.loads(anchor_path.read_text(encoding='utf-8'))
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or anchor_report['cohort_sha256']!=digest(cohort_path) or anchor_report['prepared_manifest_sha256']!=digest(prep_path):raise ValueError('Changed development data')
    lookup={e['original_path']:data/e['prepared_path'] for e in prep['files']}
    for e in prep['files']:
        if digest(lookup[e['original_path']])!=e['sha256']:raise ValueError('Changed prepared image')
    out=base/('context-appearance-v1' if args.appearance else 'context-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['train_reference_context.py','reference_context.py','reference-context-protocol.md','train_triplet_pilot.py','training_checks.py','training_gates.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,appearance=args.appearance,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),
        anchor_results_sha256=digest(anchor_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},epochs=[],finished=False,
        scope='Development only; no consumed reserved images, App replacement or semantic recognition claim')
    if args.appearance:
        control_path=base/'context-control-v1/results.json';control=json.loads(control_path.read_text(encoding='utf-8'))
        if not control['finished'] or control['config']!=config or control['cohort_sha256']!=report['cohort_sha256'] or control['prepared_manifest_sha256']!=report['prepared_manifest_sha256'] or control['start_checkpoint_sha256']!=START_SHA:raise ValueError('Mismatched control')
        control_scores_path=base/'context-control-v1'/f'epoch-{control["best_epoch"]}/scores.json'
        report['control_snapshot']=dict(results_sha256=digest(control_path),scores_sha256=digest(control_scores_path))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,_=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite weights')
    model.load_state_dict(state,strict=True);samples={}
    for part,count in [('train',512),('validation',64)]:
        samples[part]=[]
        for row in cohort['groups'][part]:
            tensors=[]
            for role in ['content','reference','gt']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                tensors.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
            samples[part].append((row,*tensors))
        if len(samples[part])!=count:raise ValueError('Incomplete sample list')
    def original(c,s):
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        raw,grid=model(c,hc,hs);finite_output(grid,(c.shape[0],13,8,16,16));return finite_output(raw,c.shape)
    oracle=[]
    if args.appearance:
        model.eval()
        with torch.inference_mode():
            for row,c,s,gt in samples['validation']:oracle.append(original(c[None].cuda(),s[None].cuda())[0].cpu())
        enable_reference_context(model);report['initial_equivalence']=[]
    initial=out/'epoch-16.pth';torch.save(model.state_dict(),initial);report['initial_checkpoint_sha256']=digest(initial)
    def forward(c,s):return reference_forward(model,api,c,s)[0] if args.appearance else original(c,s)
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();directory=out/f'epoch-{epoch}';directory.mkdir();rows=[]
        for index,(row,c,s,gt) in enumerate(samples['validation']):
            raw=forward(c[None].cuda(),s[None].cuda())[0];prediction=pixels(raw)
            if args.appearance and epoch==16:
                delta=float((raw.cpu()-oracle[index]).abs().max());byte_delta=int(np.abs(prediction.astype(int)-pixels(oracle[index]).astype(int)).max())
                report['initial_equivalence'].append(dict(index=row['index'],max_float_delta=delta,max_rgb8_delta=byte_delta))
                if delta>1e-5 or byte_delta>1:raise ValueError('Initial context equivalence failed')
            path=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            rows.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-pixels(gt)).mean()),sha256=digest(path)))
        result=dict(mean=float(np.mean([r['mae'] for r in rows])),cases=rows,subgroups=dict(original32=float(np.mean([r['mae'] for r in rows[:32]])),added32=float(np.mean([r['mae'] for r in rows[32:]]))))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate(16);best,best_epoch=baseline['mean'],16;report['baseline']=baseline
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=best)),flush=True)
    parameters=list(model.parameters())
    if args.appearance:
        head=list(model.reference_adapter.parameters());ids={id(p) for p in head}
        groups=[dict(params=[p for p in parameters if id(p) not in ids],lr=config['base_lr']),dict(params=head,lr=config['new_head_lr'])]
    else:groups=[dict(params=parameters,lr=config['base_lr'])]
    optimizer=torch.optim.AdamW(groups,weight_decay=config['weight_decay']);rng=np.random.default_rng(config['seed']);updates=0
    for epoch in range(17,25):
        model.train();losses=[];start=time.perf_counter();order=rng.permutation(512)
        for offset in range(0,len(order),4):
            batch=[samples['train'][int(j)] for j in order[offset:offset+4]];c,s,gt=[torch.stack([r[k] for r in batch]).cuda() for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);pred=forward(c,s).clamp(0,1)
            loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path);validation=evaluate(epoch)
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],checkpoint_sha256=digest(path))
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best,finished=epoch==24)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=1024 or digest(anchor_path)!=report['anchor_results_sha256']:raise ValueError('Budget or anchor changed')
    selected=json.loads((out/f'epoch-{best_epoch}/scores.json').read_text(encoding='utf-8'));comparison=None
    if args.appearance:
        if digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(control_scores_path)!=report['control_snapshot']['scores_sha256']:raise ValueError('Control changed')
        comparison=json.loads(control_scores_path.read_text(encoding='utf-8'))
    gate=development_gate(baseline,selected,True,comparison);app_gate=development_gate(anchor_report['baseline'],selected,True)
    gate['current_app_development_gate']=app_gate;gate['proceed_to_independent_evaluation'] &= app_gate['proceed_to_independent_evaluation']
    report['development_gate']=gate;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
