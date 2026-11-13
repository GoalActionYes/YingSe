"""Matched controlled-reference augmentation; development always real references."""
import argparse
import hashlib
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
from pairwise_context import enable_pairwise_context,pairwise_grid
from pairwise_gate import response_development_gate,require_completed_control
from multiref_response import response_metrics
from tone_response import tone_response
from consistent_reference import use_synthetic_reference

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'

def forward(model,api,c,s):
    grid=pairwise_grid(model,api,c,s)
    return finite_output(model.apply_coeffs(model.sampler(grid,c),c),c.shape),grid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--augment-reference',action='store_true');args=parser.parse_args()
    config=dict(seed=20261122,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    def seed():
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'multiref-control-v1/epoch-6.pth';generated_dir=base/'cohort-consistent-reference-v1';generated_path=generated_dir/'results.json'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed current weights')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep,generated=load(cohort_path),load(prep_path),load(generated_path)
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Changed cohort')
    train_rows=cohort['groups']['train'];train_ids={r['index'] for r in train_rows};train_contents={r['content'] for r in train_rows};training_images={r[k] for r in train_rows for k in ['content','reference','gt']};gen={r['index']:r for r in generated['cases']}
    if not generated['finished'] or generated['cohort_sha256']!=digest(cohort_path) or generated['prepared_manifest_sha256']!=digest(prep_path) or generated['checkpoint_sha256']!=START_SHA or len(gen)!=1024 or len(generated['cases'])!=1024 or set(gen)!=train_ids:raise ValueError('Incomplete or changed generated references')
    parent_entries={r['original_path']:r['sha256'] for r in prep['files']}
    for name,sha in generated['source_sha256'].items():
        if digest(generated_dir/'frozen-source'/name)!=sha:raise ValueError('Changed frozen generation source')
    for row in train_rows:
        item=gen[row['index']]
        parents={row['content'],row['gt'],item['reference_original']}
        if set(item['parent_sha256'])!=parents or any(parent_entries.get(p)!=sha for p,sha in item['parent_sha256'].items()):raise ValueError('Changed generation parents')
        if item['content']!=row['content'] or item['gt']!=row['gt'] or item['reference_original'] not in train_contents or item['reference_original']==row['content'] or any(p not in training_images for p in item['parent_sha256']):raise ValueError('Invalid synthetic provenance')
        path=generated_dir/item['generated_reference']
        if path.parent!=generated_dir or digest(path)!=item['sha256']:raise ValueError('Changed generated reference')
    lookup={r['original_path']:data/r['prepared_path'] for r in prep['files']}
    for row in prep['files']:
        if digest(lookup[row['original_path']])!=row['sha256']:raise ValueError('Changed prepared image')
    anchor_path=base/'multiref-control-v1/epoch-6/scores.json';anchor=load(anchor_path);anchor_rows={r['index']:r for r in anchor['cases']}
    out=base/('consistent-reference-v1' if args.augment_reference else 'consistent-reference-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['train_consistent_reference.py','consistent_reference.py','consistent-reference-protocol.md','pairwise_context.py','pairwise_gate.py','tone_response.py','paired_training.py','multiref_response.py','train_triplet_pilot.py','training_checks.py','training_gates.py']
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,augment_reference=args.augment_reference,pairwise=True,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),generated_manifest_sha256=digest(generated_path),anchor_scores_sha256=digest(anchor_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},epochs=[],finished=False,scope='Matched real-reference development; training-only controlled reference augmentation; no reserved pixels')
    comparison=None;control_dir=base/'consistent-reference-control-v1'
    if args.augment_reference:
        control_path=control_dir/'results.json';control=load(control_path);require_completed_control(control,config,report['source_sha256'])
        if control['augment_reference'] or any(control[k]!=report[k] for k in ['cohort_sha256','prepared_manifest_sha256','start_checkpoint_sha256','generated_manifest_sha256']):raise ValueError('Mismatched control')
        if digest(control_dir/f'epoch-{control["best_epoch"]}.pth')!=control['selected_checkpoint_sha256']:raise ValueError('Changed control weights')
        control_scores=control_dir/f'epoch-{control["best_epoch"]}/scores.json';comparison=load(control_scores);report['control_snapshot']=dict(results_sha256=digest(control_path),scores_sha256=digest(control_scores))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(row,*[image(lookup[row[role]]) for role in ['content','reference','gt']]) for row in cohort['groups'][part]] for part in ['train','validation']}
    synthetic={index:image(generated_dir/item['generated_reference']) for index,item in gen.items()}
    grouped_batches(train_rows,np.random.default_rng(config['seed']));model,api,author=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite initial weights')
    model.load_state_dict(state,strict=True);model.eval();report['author_revision']=author['revision']
    inputs=lambda x:(x.float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    oracle=[]
    with torch.inference_mode():
        for row,c,s,gt in samples['validation']:
            c,s=inputs(c[None]),inputs(s[None]);hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));raw,grid=model(c,hc,hs)
            oracle.append((finite_output(raw,c.shape).cpu(),finite_output(grid,(1,13,8,16,16)).cpu()))
    enable_pairwise_context(model);report['initial_full_equivalence']=[]
    if args.augment_reference:
        initial=torch.load(control_dir/'epoch-0.pth',map_location='cpu',weights_only=True);current=model.state_dict()
        if set(initial)!=set(current) or any(not torch.equal(initial[k],v.cpu()) for k,v in current.items()):raise ValueError('Initial full parameters differ')
        report['initial_state_exact_to_control']=True
    @torch.inference_mode()
    def evaluate(label):
        model.eval();directory=out/label;directory.mkdir();cases=[];groups={}
        for number,(row,c,s,gt) in enumerate(samples['validation']):
            raw,grid=forward(model,api,inputs(c[None]),inputs(s[None]));prediction=pixels(raw[0]);truth=gt.permute(1,2,0).numpy();path=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            if label=='baseline-current':
                if digest(path)!=anchor_rows[row['index']]['sha256']:raise ValueError('Current initial PNG differs')
                delta=max(float((raw.cpu()-oracle[number][0]).abs().max()),float((grid.cpu()-oracle[number][1]).abs().max()))
                if delta!=0:raise ValueError('Zero head changes initial output')
                report['initial_full_equivalence'].append(dict(index=row['index'],maximum_float_difference=delta))
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(path)));group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete evaluation')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        result=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate('baseline-current');initial=evaluate('epoch-0');report['baseline']=baseline;report['initial_output_mean']=initial['mean'];oracle.clear()
    if initial!=baseline or (args.augment_reference and baseline!=control['baseline']):raise ValueError('Initial evaluation differs')
    torch.save(model.state_dict(),out/'epoch-0.pth');report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=baseline['mean'],augment_reference=args.augment_reference)),flush=True)
    extra=list(model.pairwise_adapter.parameters());identities={id(p) for p in extra};parameters=[dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])]
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay']);seed();rng=np.random.default_rng(config['seed']);updates=0;best=initial['mean'];best_epoch=0
    for epoch in range(1,config['epochs']+1):
        model.train();losses=[];start=time.perf_counter();replaced=0;order=[]
        for indices in grouped_batches(train_rows,rng):
            batch=[samples['train'][i] for i in indices];c=inputs(torch.stack([r[1] for r in batch]));gt=inputs(torch.stack([r[3] for r in batch]));refs=[]
            for row,_,reference,_ in batch:
                use=args.augment_reference and use_synthetic_reference(row['style_index'],epoch);refs.append(synthetic[row['index']] if use else reference);replaced+=int(use);order.append(row['index'])
            s=inputs(torch.stack(refs));optimizer.zero_grad(set_to_none=True);prediction=forward(model,api,c,s)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:
                progress=dict(epoch=epoch,updates=updates,loss=float(loss));(out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
        if replaced!=(512 if args.augment_reference else 0) or len(order)!=1024 or set(order)!=train_ids:raise ValueError('Incorrect reference schedule')
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if args.augment_reference and order_sha!=control['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Training target order differs')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path);validation=evaluate(f'epoch-{epoch}')
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(path),synthetic_references=replaced,target_order_sha256=order_sha)
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=config['max_updates']:raise ValueError('Incorrect budget')
    if args.augment_reference and (digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(control_scores)!=report['control_snapshot']['scores_sha256']):raise ValueError('Control changed')
    if digest(checkpoint)!=START_SHA or digest(anchor_path)!=report['anchor_scores_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(generated_path)!=report['generated_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Starting evidence or executing sources changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');report['development_gate']=response_development_gate(baseline,selected,comparison);report['selected_checkpoint_sha256']=digest(out/f'epoch-{best_epoch}.pth');report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)

if __name__=='__main__':main()
