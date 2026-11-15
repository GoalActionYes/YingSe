"""Known-GT bounded condition fitting diagnostic; not a deployable inference."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import condition_forward
from check_petal_condition import state_hash
from multiref_response import response_metrics


def main():
    torch.set_num_threads(4);torch.manual_seed(20261126);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']};prior_dir=base/'petal-condition-training-v1';prior=load(prior_dir/'results.json')
    if not prior['finished']:raise ValueError('Prior learning incomplete')
    feature_report=load(base/'petal-condition-features-v1/results.json')
    if digest(data/'cohort.json')!=feature_report['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=feature_report['prepared_manifest_sha256']:raise ValueError('Historical cohort changed')
    rows=cohort['groups']['validation']
    if len(rows)!=64:raise ValueError('Wrong cohort')
    images={};hashes={}
    for row in rows:
        for role in ['content','reference','gt']:
            name=row[role];entry=lookup[name];path=data/entry['prepared_path']
            if digest(path)!=entry['sha256']:raise ValueError('Pixels changed')
            with Image.open(path) as im:images[name]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            hashes[name]=entry['sha256']
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=prior['checkpoint_sha256']:raise ValueError('Baseline differs')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    out=base/'condition-fit-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['probe_condition_fit.py','condition-fit-protocol.md','petal_condition.py','check_petal_condition.py','train_triplet_pilot.py','training_checks.py','multiref_response.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Known-GT per-pair condition fit; not deployable, not independent acceptance',config=dict(seed=20261126,steps=128,batch=4,lr=.03,bound=3.,gradient_weight=.1),source_sha256={n:digest(frozen/n) for n in names},prior_results_sha256=digest(prior_dir/'results.json'),cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),checkpoint_sha256=digest(cp),input_parent_hashes=hashes,gt_used_for_optimization_and_selection=True,model_weights_updated=False,product_integrated=False,backbone_state_before=before,cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def rgb(raw):return np.rint(raw.clip(0,1)*255).astype(np.uint8)
    for offset in range(0,64,4):
        batch=rows[offset:offset+4];c,s,gt=[torch.from_numpy(np.stack([images[r[k]] for r in batch])).permute(0,3,1,2).float().cuda()/255 for k in ['content','reference','gt']]
        delta=torch.nn.Parameter(torch.zeros(4,128,device='cuda'));optimizer=torch.optim.Adam([delta],lr=.03);best=[None]*4;trace=[[] for _ in batch];param_trace=[[] for _ in batch];start=time.perf_counter()
        with torch.no_grad():
            original,_=condition_forward(model,None,api,c,s,enabled=False)
            initial,_=condition_forward(model,lambda _a,_b:delta,api,c,s)
            if not torch.equal(original,initial):raise ValueError('Zero residual bridge failed')
            single=torch.cat([condition_forward(model,None,api,c[i:i+1],s[i:i+1],enabled=False)[0] for i in range(4)])
            baseline=single.cpu().permute(0,2,3,1).numpy()
        for step in range(129):
            if step%16==0:
                with torch.no_grad():
                    raw=torch.cat([condition_forward(model,lambda _a,_b:delta[i:i+1],api,c[i:i+1],s[i:i+1])[0] for i in range(4)]);array=raw.cpu().permute(0,2,3,1).numpy();params=delta.detach().cpu().numpy().copy()
                for i,row in enumerate(batch):
                    pixels=rgb(array[i]);error=float(np.abs(pixels.astype(float)-images[row['gt']].astype(float)).mean());trace[i].append(dict(step=step,mae=error));param_trace[i].append(params[i].copy())
                    if best[i] is None or error<best[i]['mae']:best[i]=dict(mae=error,step=step,raw=array[i].copy(),delta=params[i].copy())
            if step==128:break
            optimizer.zero_grad(set_to_none=True);pred,_=condition_forward(model,lambda _a,_b:delta,api,c,s);pred=pred.clamp(0,1)
            loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite fit loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([delta],1.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():delta.clamp_(-3,3)
        for i,row in enumerate(batch):
            prefix=str(row['index']);selected=best[i];files={}
            for name,pixels in [('baseline',rgb(baseline[i])),('gt_fit',rgb(selected['raw']))]:
                path=out/f'{prefix}-{name}.png';Image.fromarray(pixels).save(path);files[name]=dict(path=path.name,sha256=digest(path))
            old=next(v for v in prior['arms']['PETAL1024']['baseline']['cases'] if v['index']==row['index'])
            if files['baseline']['sha256']!=old['sha256']:raise ValueError('Historical baseline changed')
            cache=out/f'{prefix}-fit.npz';np.savez_compressed(cache,baseline_raw=baseline[i],selected_raw=selected['raw'],selected_delta=selected['delta'],snapshot_deltas=np.stack(param_trace[i]),snapshot_steps=np.arange(0,129,16));files['raw']=dict(path=cache.name,sha256=digest(cache))
            report['cases'].append(dict(index=row['index'],group_id=row['group_id'],style_index=row['style_index'],baseline_mae=trace[i][0]['mae'],fit_mae=selected['mae'],selected_step=selected['step'],delta_max_abs=float(np.abs(selected['delta']).max()),snapshots=trace[i],files=files))
        print(json.dumps(dict(done=offset+4,total=64,batch_seconds=time.perf_counter()-start)),flush=True)
    groups={}
    for row in report['cases']:groups.setdefault(row['group_id'],{})[row['style_index']]=row
    paired=[]
    for group,items in groups.items():
        pair={}
        for name in ['baseline','gt_fit']:
            a,b=[np.asarray(Image.open(out/items[i]['files'][name]['path']).convert('RGB')) for i in [0,1]];ga,gb=[images[next(r['gt'] for r in rows if r['index']==items[i]['index'])] for i in [0,1]];pair[name]=response_metrics(a,b,ga,gb)
        paired.append(dict(group_id=group,metrics=pair))
    a=np.array([r['baseline_mae'] for r in report['cases']]);b=np.array([r['fit_mae'] for r in report['cases']]);report['summary']=dict(baseline_mean=float(a.mean()),fit_mean=float(b.mean()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()),gain=float(1-b.mean()/a.mean()),further_condition_prediction_research_value=bool(b.mean()<=.95*a.mean() and (b<a-.1).sum()>=40));report['paired']=paired;report['backbone_state_after']=state_hash(model)
    if report['backbone_state_after']!=before or any(p.grad is not None for p in model.parameters()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen evidence/model changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
