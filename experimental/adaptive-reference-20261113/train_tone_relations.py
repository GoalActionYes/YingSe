"""Matched pairwise-model training with optional fixed-target tone supervision."""
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
from paired_training import grouped_batches
from pairwise_context import enable_pairwise_context,pairwise_forward
from pairwise_gate import response_development_gate,require_completed_control
from multiref_response import response_metrics
from tone_response import tone_response
from tone_relations import tone_relation_losses

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--relations',action='store_true');args=parser.parse_args()
    config=dict(seed=20261119,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.,absolute_tone_weight=.25,paired_tone_weight=.5)
    def seed():
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed current starting weights')
    numeric_path=ROOT/'outputs/adaptive-reference-20261113/tone-relations/numeric-v2/results.json';numeric=json.loads(numeric_path.read_text(encoding='utf-8'))
    if not numeric['passed'] or numeric['source_sha256']['tone_relations.py']!=digest(Path(__file__).with_name('tone_relations.py')) or numeric['source_sha256']['tone-relations-protocol.md']!=digest(Path(__file__).with_name('tone-relations-protocol.md')):raise ValueError('Numeric checks do not cover current loss')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    anchor_path=base/'multiref-control-v1/epoch-6/scores.json';anchor=json.loads(anchor_path.read_text(encoding='utf-8'));anchor_rows={r['index']:r for r in anchor['cases']}
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Changed grouped cohort')
    lookup={e['original_path']:data/e['prepared_path'] for e in prep['files']}
    for e in prep['files']:
        if digest(lookup[e['original_path']])!=e['sha256']:raise ValueError('Changed prepared image')
    out=base/('tone-relations-v1' if args.relations else 'tone-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['train_tone_relations.py','tone_relations.py','tone-relations-protocol.md','pairwise_context.py','pairwise_gate.py','tone_response.py','paired_training.py','multiref_response.py','train_triplet_pilot.py','training_checks.py','training_gates.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,relations=args.relations,pairwise=True,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),numeric_checks_sha256=digest(numeric_path),
        anchor_scores_sha256=digest(anchor_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},epochs=[],finished=False,scope='Matched development tone-loss study; no reserved inputs or App promotion')
    comparison=None;control_dir=base/'tone-control-v1'
    if args.relations:
        control_path=control_dir/'results.json';control=json.loads(control_path.read_text(encoding='utf-8'));require_completed_control(control,config,report['source_sha256'])
        if control['relations'] or control['cohort_sha256']!=report['cohort_sha256'] or control['prepared_manifest_sha256']!=report['prepared_manifest_sha256'] or control['start_checkpoint_sha256']!=START_SHA or control['numeric_checks_sha256']!=report['numeric_checks_sha256']:raise ValueError('Mismatched ordinary control')
        if digest(control_dir/f'epoch-{control["best_epoch"]}.pth')!=control['selected_checkpoint_sha256']:raise ValueError('Changed control checkpoint')
        control_scores=control_dir/f'epoch-{control["best_epoch"]}/scores.json';comparison=json.loads(control_scores.read_text(encoding='utf-8'))
        report['control_snapshot']=dict(results_sha256=digest(control_path),scores_sha256=digest(control_scores))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    samples={}
    for part in ['train','validation']:
        samples[part]=[]
        for row in cohort['groups'][part]:
            values=[]
            for role in ['content','reference','gt']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                values.append(torch.from_numpy(rgb.copy()).permute(2,0,1))
            samples[part].append((row,*values))
    train_rows=[s[0] for s in samples['train']];grouped_batches(train_rows,np.random.default_rng(config['seed']))
    model,api,author=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite starting weights')
    model.load_state_dict(state,strict=True);model.eval();report['author_revision']=author['revision']
    def inputs(value):return (value.float()/255).cuda()
    def original(c,s):
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));raw,grid=model(c,hc,hs)
        return finite_output(raw,c.shape),finite_output(grid,(c.shape[0],13,8,16,16))
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    oracle=[]
    with torch.inference_mode():
        for row,c,s,gt in samples['validation']:
            raw,grid=original(inputs(c[None]),inputs(s[None]));oracle.append((raw.cpu(),grid.cpu()))
    enable_pairwise_context(model);report['initial_equivalence']=[];report['additional_parameters']=sum(p.numel() for p in model.pairwise_adapter.parameters())
    if args.relations:
        initial=torch.load(control_dir/'epoch-0.pth',map_location='cpu',weights_only=True);current=model.state_dict()
        if set(initial)!=set(current) or any(not torch.equal(initial[k],v.cpu()) for k,v in current.items()):raise ValueError('Initial full model differs from control')
        report['initial_state_exact_to_control']=True
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();directory=out/f'epoch-{epoch}';directory.mkdir();cases=[];groups={}
        for number,(row,c,s,gt) in enumerate(samples['validation']):
            raw,grid=pairwise_forward(model,api,inputs(c[None]),inputs(s[None]));prediction=pixels(raw[0]);truth=gt.permute(1,2,0).numpy();path=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            if epoch==0:
                if digest(path)!=anchor_rows[row['index']]['sha256']:raise ValueError('Starting PNG differs from current model')
                delta=max(float((raw.cpu()-oracle[number][0]).abs().max()),float((grid.cpu()-oracle[number][1]).abs().max()))
                if delta!=0:raise ValueError('Zero head changes initial floating output')
                report['initial_equivalence'].append(dict(index=row['index'],maximum_float_difference=delta))
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(path)));group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete paired evaluation')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        result=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate(0);oracle.clear();report['baseline']=baseline;torch.save(model.state_dict(),out/'epoch-0.pth');report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth')
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=baseline['mean'],delta=baseline['delta_mean'],relations=args.relations)),flush=True)
    extra=list(model.pairwise_adapter.parameters());identities={id(p) for p in extra};parameters=[dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])]
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay']);seed();rng=np.random.default_rng(config['seed']);updates=0;best=baseline['mean'];best_epoch=0
    for epoch in range(1,config['epochs']+1):
        model.train();losses=[];tone_losses=[];start=time.perf_counter()
        for indices in grouped_batches(train_rows,rng):
            batch=[samples['train'][i] for i in indices];c,s,gt=[inputs(torch.stack([r[k] for r in batch])) for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);prediction=pairwise_forward(model,api,c,s)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if args.relations:
                absolute,paired=tone_relation_losses(prediction,gt,c);loss=loss+.25*absolute+.5*paired;tone_losses.append([float(absolute),float(paired)])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%32==0:
                progress=dict(epoch=epoch,updates=updates,loss=float(loss));(out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path);validation=evaluate(epoch)
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),tone_loss_means=np.mean(tone_losses,axis=0).tolist() if tone_losses else None,training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(path))
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=config['max_updates']:raise ValueError('Unexpected training budget')
    if args.relations and (digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(control_scores)!=report['control_snapshot']['scores_sha256']):raise ValueError('Control changed')
    if digest(checkpoint)!=START_SHA or digest(anchor_path)!=report['anchor_scores_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256']:raise ValueError('Starting evidence changed')
    selected=json.loads((out/f'epoch-{best_epoch}/scores.json').read_text(encoding='utf-8'));report['development_gate']=response_development_gate(baseline,selected,comparison)
    report['selected_checkpoint_sha256']=digest(out/f'epoch-{best_epoch}.pth');report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)


if __name__=='__main__':main()
