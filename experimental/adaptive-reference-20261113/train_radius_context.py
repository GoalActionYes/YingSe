"""One fixed normalized-appearance candidate, completed C10 is matched control."""
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
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from radius_context import radius_forward,calibrated_radius
from paired_training import grouped_batches
from pairwise_gate import require_completed_control,response_development_gate
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';control=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));record=load(control/'results.json');cal_path=base/'context-radius-calibration-v1/results.json';cal=load(cal_path)
    config=dict(seed=20261124,appearance_init_seed=20261125,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    require_completed_control(record,config,record['source_sha256'])
    if not record['appearance_enabled'] or record['best_epoch']!=7 or digest(control/'results.json')!='3655c1120a7d854269304810c3be144e281d871a4cd6d1cd62c2236e2b7d093c':raise ValueError('Wrong matched C10 control')
    for name,sha in record['source_sha256'].items():
        if digest(control/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Control source changed')
    if not cal['finished'] or len(cal['inputs'])!=2048 or cal['radius']!=calibrated_radius([v['norm'] for v in cal['inputs']]):raise ValueError('Calibration incomplete')
    for name,sha in cal['source_sha256'].items():
        if digest(cal_path.parent/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Calibration source changed')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);checkpoint=base/'multiref-control-v1/epoch-6.pth';radius=cal['radius']
    if digest(cohort_path)!=record['cohort_sha256'] or digest(prep_path)!=record['prepared_manifest_sha256'] or digest(checkpoint)!=record['start_checkpoint_sha256'] or any(cal[k]!=record[k] for k in ['cohort_sha256','prepared_manifest_sha256']) or cal['checkpoint_sha256']!=record['start_checkpoint_sha256']:raise ValueError('Calibration/control data differs')
    lookup={v['original_path']:data/v['prepared_path'] for v in prep['files']}
    for v in prep['files']:
        if digest(lookup[v['original_path']])!=v['sha256']:raise ValueError('Prepared asset changed')
    scores_path=control/'epoch-7/scores.json';weight_path=control/'epoch-7.pth';initial_path=control/'epoch-0.pth';comparison=load(scores_path)
    if digest(weight_path)!=record['selected_checkpoint_sha256'] or digest(initial_path)!=record['initial_checkpoint_sha256']:raise ValueError('Control weights changed')
    snapshot=dict(results_sha256=digest(control/'results.json'),scores_sha256=digest(scores_path),selected_sha256=digest(weight_path),initial_sha256=digest(initial_path));out=base/'radius-context-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=set(record['source_sha256'])|set(cal['source_sha256'])|{'train_radius_context.py','audit_consistent_reference.py'}
    for name in sorted(names):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='One radius-normalized real-reference development candidate; no G64 feature use or independent acceptance',config=config,radius=radius,calibration_results_sha256=digest(cal_path),control_snapshot=snapshot,source_sha256={p.name:digest(p) for p in frozen.iterdir()},cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),start_checkpoint_sha256=digest(checkpoint),epochs=[],initial_raw_equivalence=[],reused_control_native=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def seed():random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();model,api,author=author_model();initialize_replication_model(model,torch.load(checkpoint,map_location='cpu',weights_only=True));initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    if set(initial)!=set(model.state_dict()) or any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('Complete initial state differs')
    report.update(initial_complete_state_exact_to_C10=True,author_revision=author['revision'])
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(r,*[image(lookup[r[k]]) for k in ['content','reference','gt']]) for r in cohort['groups'][part]] for part in ['train','validation']};rows=cohort['groups']['validation'];truth={r['index']:gt.permute(1,2,0).numpy() for r,c,s,gt in samples['validation']};inputs=lambda v:(v.float()/255).cuda();pixels=lambda x:np.rint(x[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(label,normalized=True):
        model.eval();directory=out/label;directory.mkdir();predictions={}
        for row,c,s,gt in samples['validation']:
            c,s=inputs(c[None]),inputs(s[None]);raw,grid=radius_forward(model,api,c,s,radius,normalized);i=row['index'];predictions[i]=pixels(raw);Image.fromarray(predictions[i]).save(directory/f'{i}.png')
            if label in ['reused-control','epoch-0']:
                old,old_grid=appearance_forward(model,api,c,s);difference=max(float((raw-old).abs().max()),float((grid-old_grid).abs().max()))
                if difference!=0:raise ValueError('Native raw/grid bridge differs')
                report['reused_control_native' if label=='reused-control' else 'initial_raw_equivalence'].append(dict(index=i,maximum_float_difference=difference))
        scores=score_outputs(rows,predictions,truth)
        for v in scores['cases']:v['sha256']=digest(directory/f'{v["index"]}.png')
        (directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');return scores
    model.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True),strict=True)
    if evaluate('reused-control',normalized=False)!=comparison:raise ValueError('Matched C10 native RGB/full metrics differ')
    model.load_state_dict(initial,strict=True)
    if any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('Initial restoration failed')
    baseline=evaluate('epoch-0')
    if baseline!=record['baseline']:raise ValueError('Initial current13 PNG/metrics differ')
    report['baseline']=baseline;(out/'epoch-0.pth').write_bytes(initial_path.read_bytes());report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(initial_exact=True,matched_C10_native_exact=True,radius=radius)),flush=True)
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());identities={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay']);seed();rng=np.random.default_rng(config['seed']);updates=0;best=baseline['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];order=[];start=time.perf_counter()
        for indices in grouped_batches(cohort['groups']['train'],rng):
            batch=[samples['train'][i] for i in indices];c,s,gt=[inputs(torch.stack([v[k] for v in batch])) for k in [1,2,3]];order.extend(v[0]['index'] for v in batch);optimizer.zero_grad(set_to_none=True);prediction=radius_forward(model,api,c,s,radius)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:print(json.dumps(dict(epoch=epoch,updates=updates,loss=float(loss))),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if len(order)!=1024 or len(set(order))!=1024 or order_sha!=record['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Training order differs')
        torch.cuda.synchronize();seconds=time.perf_counter()-start;p=out/f'epoch-{epoch}.pth'
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite epoch state')
        torch.save(model.state_dict(),p);scores=evaluate(f'epoch-{epoch}')
        if scores['mean']<best:best,best_epoch=scores['mean'],epoch
        item=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=scores['mean'],delta_mean=scores['delta_mean'],checkpoint_sha256=digest(p),target_order_sha256=order_sha);report['epochs'].append(item);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)
    if updates!=2048 or digest(cal_path)!=report['calibration_results_sha256'] or digest(control/'results.json')!=snapshot['results_sha256'] or digest(scores_path)!=snapshot['scores_sha256'] or digest(weight_path)!=snapshot['selected_sha256'] or digest(initial_path)!=snapshot['initial_sha256'] or digest(checkpoint)!=report['start_checkpoint_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen evidence or budget changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');report.update(development_gate=response_development_gate(baseline,selected,comparison),selected_checkpoint_sha256=digest(out/f'epoch-{best_epoch}.pth'),finished=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)


if __name__=='__main__':main()
