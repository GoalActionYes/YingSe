"""Matched grouped supervision, optionally adding fixed reference-delta loss."""
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
from paired_training import grouped_batches,paired_delta_loss
from multiref_response import response_metrics

START_SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');args=parser.parse_args()
    config=dict(seed=20261117,lr=1e-5,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,delta_weight=.25,gradient_clip=1.)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'training-v1/epoch-8.pth'
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed App starting weights')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Incomplete grouped cohort')
    lookup={e['original_path']:data/e['prepared_path'] for e in prep['files']}
    for e in prep['files']:
        if digest(lookup[e['original_path']])!=e['sha256']:raise ValueError('Changed prepared image')
    out=base/('multiref-paired-v1' if args.paired else 'multiref-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['train_multiref.py','paired_training.py','multiref_response.py','multiref-training-protocol.md','train_triplet_pilot.py','training_checks.py','training_gates.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,paired=args.paired,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},epochs=[],finished=False,scope='Grouped development study, no independent evaluation or App promotion')
    control=None
    if args.paired:
        control_path=base/'multiref-control-v1/results.json';control=json.loads(control_path.read_text(encoding='utf-8'))
        if not control['finished'] or control['config']!=config or control['cohort_sha256']!=report['cohort_sha256'] or control['prepared_manifest_sha256']!=report['prepared_manifest_sha256'] or control['start_checkpoint_sha256']!=START_SHA:raise ValueError('Mismatched control')
        control_scores=base/'multiref-control-v1'/f'epoch-{control["best_epoch"]}/scores.json'
        report['control_snapshot']=dict(results_sha256=digest(control_path),scores_sha256=digest(control_scores))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    samples={}
    for part in ['train','validation']:
        samples[part]=[]
        for row in cohort['groups'][part]:
            tensors=[]
            for role in ['content','reference','gt']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                tensors.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
            samples[part].append((row,*tensors))
    train_rows=[s[0] for s in samples['train']];grouped_batches(train_rows,np.random.default_rng(config['seed']))
    model,api,_=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite weights')
    model.load_state_dict(state,strict=True)
    def forward(c,s):
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        raw,grid=model(c,hc,hs);finite_output(grid,(c.shape[0],13,8,16,16));return finite_output(raw,c.shape)
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();directory=out/f'epoch-{epoch}';directory.mkdir();cases=[];groups={}
        for row,c,s,gt in samples['validation']:
            prediction=pixels(forward(c[None].cuda(),s[None].cuda())[0]);truth=pixels(gt);path=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(path)))
            group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate development style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete development pairs')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        result=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate(0);report['baseline']=baseline;torch.save(model.state_dict(),out/'epoch-0.pth')
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=baseline['mean'],delta=baseline['delta_mean'])),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);rng=np.random.default_rng(config['seed']);updates=0;best=baseline['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];start=time.perf_counter();batches=grouped_batches(train_rows,rng)
        for indices in batches:
            batch=[samples['train'][i] for i in indices];c,s,gt=[torch.stack([r[k] for r in batch]).cuda() for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);prediction=forward(c,s).clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if args.paired:loss=loss+.25*paired_delta_loss(prediction,gt)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path);validation=evaluate(epoch)
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(path))
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best,finished=epoch==8)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=2048:raise ValueError('Unexpected training budget')
    selected=json.loads((out/f'epoch-{best_epoch}/scores.json').read_text(encoding='utf-8'));comparison=None
    if args.paired:
        if digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(control_scores)!=report['control_snapshot']['scores_sha256']:raise ValueError('Control changed')
        comparison=json.loads(control_scores.read_text(encoding='utf-8'))
    gate=development_gate(baseline,selected,True,comparison)
    gate['delta_improvement_vs_app']=1-selected['delta_mean']/baseline['delta_mean'] if baseline['delta_mean']>0 else 0.
    gate['proceed_to_independent_evaluation'] &= gate['delta_improvement_vs_app']>=.05
    if comparison is not None:
        before={r['group_id']:r['delta_mae'] for r in comparison['paired']};after={r['group_id']:r['delta_mae'] for r in selected['paired']}
        if len(before)!=32 or set(before)!=set(after):raise ValueError('Paired control identities differ')
        gate['delta_improvement_vs_control']=1-selected['delta_mean']/comparison['delta_mean'] if comparison['delta_mean']>0 else 0.
        gate['delta_groups_improved_vs_control']=sum(before[k]-after[k]>.1 for k in before)
        gate['proceed_to_independent_evaluation'] &= gate['delta_improvement_vs_control']>=.05 and gate['delta_groups_improved_vs_control']>=20
    report['development_gate']=gate;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
