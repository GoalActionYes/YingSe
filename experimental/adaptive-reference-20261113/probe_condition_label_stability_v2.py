"""Train-only multistart condition-label diagnosis; no product inference."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import condition_forward
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash


def main():
    torch.set_num_threads(4);torch.manual_seed(20261126);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');features=load(base/'petal-condition-features-v1/results.json');bridge_path=base/'prepared-condition-bridge-v2/results.json';bridge=load(bridge_path)
    if not bridge['finished'] or digest(data/'cohort.json')!=features['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=features['prepared_manifest_sha256']:raise ValueError('Data/bridge changed')
    if any(v['output']!=0 or v['grid']!=0 or not v['gradient_allclose'] for c in bridge['cases'] for v in c['exact_differences']):raise ValueError('Cache bridge failed')
    for n,sha in bridge['source_sha256'].items():
        if digest(Path(__file__).with_name(n))!=sha:raise ValueError('Cache bridge source changed')
    rows=cohort['groups']['train'];groups={}
    for row in rows:groups.setdefault(row['group_id'],[]).append(row)
    if len(rows)!=1024 or len(groups)!=256 or any(sorted(r['style_index'] for r in g)!=list(range(4)) for g in groups.values()):raise ValueError('Wrong training split')
    rows=rows[:32];teacher_dir=base/'condition-labels-train-v1';teachers=load(teacher_dir/'results.json');teacher_audit=load(teacher_dir/'artifact-audit.json')
    if not teachers['finished'] or teacher_audit['producer_results_sha256']!=digest(teacher_dir/'results.json'):raise ValueError('Teacher audit changed')
    lookup={v['original_path']:v for v in prep['files']};out=base/'condition-label-stability-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['probe_condition_label_stability_v2.py','condition-label-stability-v2-protocol.md','prepared_condition.py','petal_condition.py','train_triplet_pilot.py','check_petal_condition.py','training_checks.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=bridge['checkpoint_sha256']:raise ValueError('Wrong backbone')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    report=dict(scope='First32 training rows multistart known-GT diagnosis, not inference or representative acceptance',split='train',config=dict(seed=20261126,perturbation_seed=20261128,initial_std=.5,steps=128,batch=4,lr=.03,bound=3.,gradient_weight=.1),source_sha256={n:digest(frozen/n) for n in names},cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),checkpoint_sha256=digest(cp),bridge_sha256=digest(bridge_path),backbone_state_before=before,gt_used_for_labels=True,teacher_results_sha256=digest(teacher_dir/'results.json'),development_labels_used=False,model_weights_updated=False,product_integrated=False,cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def pixels(raw):return np.rint(raw.clip(0,1)*255).astype(np.uint8)
    initials=np.random.default_rng(20261128).normal(0,.5,(32,128)).astype(np.float32)
    for arm,initial in [('zero',np.zeros_like(initials)),('zero-repeat',np.zeros_like(initials)),('positive',initials),('negative',-initials)]:
        arm_dir=out/arm;arm_dir.mkdir()
        for offset in range(0,32,4):
            start=time.perf_counter();batch=rows[offset:offset+4];arrays={};input_records=[]
            for row in batch:
                hashes={}
                for role in ['content','reference','gt']:
                    entry=lookup[row[role]];path=data/entry['prepared_path']
                    if digest(path)!=entry['sha256']:raise ValueError('Training file changed')
                    with Image.open(path) as im:arrays[row[role]]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                    hashes[role]=dict(path=row[role],parent_sha256=entry['sha256'])
                input_records.append(hashes)
            c,s,gt=[torch.from_numpy(np.stack([arrays[r[k]] for r in batch])).permute(0,3,1,2).float().cuda()/255 for k in ['content','reference','gt']]
            cached=PreparedCondition(model,api,c,s);singles=[PreparedCondition(model,api,c[i:i+1],s[i:i+1]) for i in range(4)];delta=torch.nn.Parameter(torch.from_numpy(initial[offset:offset+4].copy()).cuda());optimizer=torch.optim.Adam([delta],lr=.03)
            with torch.no_grad():
                original,grid=condition_forward(model,lambda _a,_b:delta,api,c,s);zero,zgrid=cached(delta)
                if not torch.equal(original,zero) or not torch.equal(grid,zgrid):raise ValueError('Cached batch baseline mismatch')
                for i in range(4):
                    expected,_=condition_forward(model,lambda _a,_b:delta[i:i+1],api,c[i:i+1],s[i:i+1]);actual,_=singles[i](delta[i:i+1])
                    if not torch.equal(expected,actual):raise ValueError('Cached single baseline mismatch')
            best=[None]*4;traces=[[] for _ in batch];parameters=[[] for _ in batch];baselines=[None]*4
            for step in range(129):
                if step%16==0:
                    with torch.no_grad():
                        snapshot=[singles[i](delta[i:i+1])[0][0].cpu().permute(1,2,0).numpy() for i in range(4)];params=delta.detach().cpu().numpy().copy()
                    for i,row in enumerate(batch):
                        image=pixels(snapshot[i]);error=float(np.abs(image.astype(float)-arrays[row['gt']].astype(float)).mean());traces[i].append(dict(step=step,mae=error));parameters[i].append(params[i].copy())
                        if step==0:baselines[i]=image.copy()
                        if best[i] is None or error<best[i]['mae']:best[i]=dict(step=step,mae=error,image=image.copy(),delta=params[i].copy())
                if step==128:break
                optimizer.zero_grad(set_to_none=True);pred=cached(delta)[0].clamp(0,1);loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
                if not torch.isfinite(loss):raise ValueError('Invalid target-fit loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([delta],1.,error_if_nonfinite=True);optimizer.step()
                with torch.no_grad():delta.clamp_(-3,3)
            for i,row in enumerate(batch):
                prefix=str(row['index']);files={}
                for name,image in [('baseline',baselines[i]),('gt_fit',best[i]['image'])]:
                    p=arm_dir/f'{prefix}-{name}.png';Image.fromarray(image).save(p);files[name]=dict(path=p.relative_to(out).as_posix(),sha256=digest(p))
                p=arm_dir/f'{prefix}-conditions.npz';np.savez_compressed(p,selected_delta=best[i]['delta'],snapshot_deltas=np.stack(parameters[i]),snapshot_steps=np.arange(0,129,16));files['conditions']=dict(path=p.relative_to(out).as_posix(),sha256=digest(p))
                if arm in ['zero','zero-repeat']:
                    old=teachers['cases'][offset+i]
                    if old['index']!=row['index'] or files['baseline']['sha256']!=old['files']['baseline']['sha256'] or best[i]['step']!=old['selected_step']:raise ValueError('Historical zero baseline/step parity failed')
                    with Image.open(teacher_dir/old['files']['gt_fit']['path']) as im:historic=np.array(im).astype(float)
                    difference=np.abs(historic-best[i]['image'].astype(float))
                    with np.load(teacher_dir/old['files']['conditions']['path']) as z:distance=float(np.sqrt(np.mean((best[i]['delta'].astype(float)-z['selected_delta'].astype(float))**2)))
                    parity=dict(arm=arm,index=row['index'],output_mae=float(difference.mean()),output_max=float(difference.max()),parameter_rms=distance)
                    if difference.mean()>.01 or difference.max()>1 or distance>.001:raise ValueError('Historical zero numeric parity failed')
                    report.setdefault('historical_zero_parity',[]).append(parity)
                report['cases'].append(dict(arm=arm,index=row['index'],group_id=row['group_id'],style_index=row['style_index'],inputs=input_records[i],baseline_mae=traces[i][0]['mae'],fit_mae=best[i]['mae'],selected_step=best[i]['step'],snapshots=traces[i],files=files))
            (out/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(dict(arm=arm,done=offset+4,total=32,batch_seconds=time.perf_counter()-start)),flush=True)
    report['backbone_state_after']=state_hash(model)
    if before!=report['backbone_state_after'] or any(p.grad is not None for p in model.parameters()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen model/source changed')
    grouped={arm:[c for c in report['cases'] if c['arm']==arm] for arm in ['zero','zero-repeat','positive','negative']};comparisons={}
    for arm in ['zero-repeat','positive','negative']:
        records=[]
        for zero,other in zip(grouped['zero'],grouped[arm]):
            with np.load(out/zero['files']['conditions']['path']) as z:zd=z['selected_delta'].astype(float)
            with np.load(out/other['files']['conditions']['path']) as z:od=z['selected_delta'].astype(float)
            with Image.open(out/zero['files']['gt_fit']['path']) as im:zi=np.array(im).astype(float)
            with Image.open(out/other['files']['gt_fit']['path']) as im:oi=np.array(im).astype(float)
            distance=float(np.sqrt(np.mean((od-zd)**2)));image_distance=float(np.abs(oi-zi).mean());score_difference=other['fit_mae']-zero['fit_mae']
            records.append(dict(index=zero['index'],parameter_rms=distance,output_rgb_mae=image_distance,output_max_abs=float(np.abs(oi-zi).max()),gt_mae_difference=score_difference,near_output=bool(distance>=.25 and image_distance<=1 and score_difference<=1),near_objective=bool(distance>=.25 and abs(score_difference)<=.25)))
        if arm=='zero-repeat' and any(c['parameter_rms']>.001 or c['output_rgb_mae']>.01 or c['output_max_abs']>1 for c in records):raise ValueError('Repeated zero numeric parity failed')
        comparisons[arm]=records
    report['comparisons']=comparisons;report['summary']={arm:dict(count=len(cases),initial_mean=float(np.mean([c['baseline_mae'] for c in cases])),fit_mean=float(np.mean([c['fit_mae'] for c in cases]))) for arm,cases in grouped.items()}
    report['diagnosis']={arm:dict(near_output=sum(c['near_output'] for c in cases),near_objective=sum(c['near_objective'] for c in cases)) for arm,cases in comparisons.items() if arm!='zero-repeat'};report['near_output_evidence']=bool(any(v['near_output']>=8 for v in report['diagnosis'].values()));report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
