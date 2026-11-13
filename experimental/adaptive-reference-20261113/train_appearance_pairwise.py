"""Pooled appearance + pairwise conditioning against a validated reused control."""
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
from pairwise_context import enable_pairwise_context
from appearance_pairwise import enable_appearance_context,appearance_forward
from pairwise_gate import require_completed_control,response_development_gate
from paired_training import grouped_batches
from training_checks import finite_output
from multiref_response import response_metrics
from tone_response import tone_response


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';control_dir=base/'consistent-reference-control-v1';bridge_dir=base/'appearance-pairwise-bridge-cpu-v3'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));control_path=control_dir/'results.json';bridge_path=bridge_dir/'results.json';control,bridge=load(control_path),load(bridge_path);config=control['config']
    require_completed_control(control,config,control['source_sha256'])
    if control['augment_reference'] or config!=dict(seed=20261122,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.):raise ValueError('Unexpected reused control')
    control_scores=control_dir/f'epoch-{control["best_epoch"]}/scores.json';comparison=load(control_scores)
    expected=dict(results_sha256=digest(control_path),selected_scores_sha256=digest(control_scores),initial_checkpoint_sha256=digest(control_dir/'epoch-0.pth'),selected_checkpoint_sha256=digest(control_dir/f'epoch-{control["best_epoch"]}.pth'))
    step=bridge['one_step_bridge']
    if not bridge['finished'] or not bridge['passed'] or bridge['control_snapshot']!=expected or not bridge['shared_initial_parameters_identical'] or step['device']!='cpu' or step['shared_parameter_maximum_difference']!=0 or step['control']!=step['disabled_appearance'] or not step['new_parameter_gradients_absent']:raise ValueError('Incomplete control reuse bridge')
    for label in ['initial_native','selected_control_native']:
        if len(bridge[label])!=64 or any(r['raw_grid_maximum_difference']!=0 or r['maximum_code_difference']!=0 for r in bridge[label]):raise ValueError('Invalid bridge native record')
    for name,sha in bridge['source_sha256'].items():
        if digest(bridge_dir/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed bridge execution source')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path)
    if digest(cohort_path)!=control['cohort_sha256'] or digest(prep_path)!=control['prepared_manifest_sha256'] or not prep['passed'] or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Changed data')
    ids={r['index'] for r in cohort['groups']['validation']}
    if any({r['index'] for r in bridge[label]}!=ids for label in ['initial_native','selected_control_native']):raise ValueError('Wrong bridge cases')
    lookup={r['original_path']:data/r['prepared_path'] for r in prep['files']}
    for r in prep['files']:
        if digest(lookup[r['original_path']])!=r['sha256']:raise ValueError('Changed prepared image')
    init_path=bridge_dir/'epoch-0.pth'
    if digest(init_path)!=bridge['candidate_initial_checkpoint_sha256']:raise ValueError('Changed bridged initial state')
    out=base/'appearance-pairwise-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=sorted(set(['train_appearance_pairwise.py',*bridge['source_sha256']]))
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Single-seed multi-reference development; reused completed real-reference control validated by CPU/CUDA bridge; not independent or product acceptance',config=config,control_reused=True,control_snapshot=expected,bridge_results_sha256=digest(bridge_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),start_checkpoint_sha256=control['start_checkpoint_sha256'],new_appearance_parameters=65792,baseline=control['baseline'],epochs=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def seed():
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();model,api,author=author_model();enable_pairwise_context(model);enable_appearance_context(model)
    state=torch.load(init_path,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite initial state')
    model.load_state_dict(state,strict=True);report['author_revision']=author['revision']
    common=torch.load(control_dir/'epoch-0.pth',map_location='cpu',weights_only=True)
    if set(state)-set(common)!={f'reference_adapter.layers.{i}.{field}' for i in [0,2] for field in ['weight','bias']} or any(not torch.equal(v,state[k]) for k,v in common.items()):raise ValueError('Shared initial state differs')
    (out/'epoch-0.pth').write_bytes(init_path.read_bytes());report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');report['shared_initial_state_exact_to_control']=True
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(row,*[image(lookup[row[role]]) for role in ['content','reference','gt']]) for row in cohort['groups'][part]] for part in ['train','validation']}
    inputs=lambda x:(x.float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(label):
        model.eval();directory=out/label;directory.mkdir();cases=[];groups={}
        for row,c,s,gt in samples['validation']:
            prediction=pixels(appearance_forward(model,api,inputs(c[None]),inputs(s[None]))[0][0]);truth=gt.permute(1,2,0).numpy();p=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(p)
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(p)));group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate evaluation style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete real-reference evaluation')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        result=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate('baseline-current');initial=evaluate('epoch-0')
    if baseline!=control['baseline'] or initial!=baseline:raise ValueError('Initial real-reference outputs differ from current13')
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=baseline['mean'],control_reused=True,shared_initial_exact=True)),flush=True)
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());identities={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    seed();rng=np.random.default_rng(config['seed']);train_rows=cohort['groups']['train'];updates=0;best=initial['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];order=[];start=time.perf_counter()
        for indices in grouped_batches(train_rows,rng):
            batch=[samples['train'][i] for i in indices];c,s,gt=[inputs(torch.stack([r[k] for r in batch])) for k in [1,2,3]];order.extend(r[0]['index'] for r in batch)
            optimizer.zero_grad(set_to_none=True);prediction=appearance_forward(model,api,c,s)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:
                progress=dict(epoch=epoch,updates=updates,loss=float(loss));(out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if len(order)!=1024 or set(order)!={r['index'] for r in train_rows} or order_sha!=control['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Training target stream differs')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained parameters')
        p=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),p);validation=evaluate(f'epoch-{epoch}')
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        row=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(p),target_order_sha256=order_sha)
        report['epochs'].append(row);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    if updates!=2048 or digest(control_path)!=expected['results_sha256'] or digest(control_scores)!=expected['selected_scores_sha256'] or digest(bridge_path)!=report['bridge_results_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Budget or frozen evidence changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');report['development_gate']=response_development_gate(baseline,selected,comparison);report['selected_checkpoint_sha256']=digest(out/f'epoch-{best_epoch}.pth');report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)

if __name__=='__main__':main()
