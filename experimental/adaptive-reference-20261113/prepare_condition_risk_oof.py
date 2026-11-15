"""Train-only target-group OOF color corrections; no development GT or gate tuning."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={r['original_path']:r for r in prep['files']};rows=cohort['groups']['train'];meta=load(base/'petal-condition-features-v1/results.json');records=[r for r in meta['rows'] if r['split']=='train'];labels_dir=base/'condition-labels-train-v1';labels=load(labels_dir/'results.json');audit=load(labels_dir/'artifact-audit.json');context_dir=base/'condition-context-ridge-v1';context_report=load(context_dir/'results.json');hdir=base/'petal-condition-training-v1';hreport=load(hdir/'results.json')
    if len(rows)!=1024 or not labels['finished'] or audit['producer_results_sha256']!=digest(labels_dir/'results.json') or audit['condition_snapshots_checked']!=9216 or digest(data/'cohort.json')!=meta['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=meta['prepared_manifest_sha256']:raise ValueError('Data lineage changed')
    if digest(context_dir/'observable-contexts.npz')!=context_report['observable_contexts_sha256'] or digest(hdir/'control-features.npz')!=hreport['control_features_sha256']:raise ValueError('Features changed')
    with np.load(context_dir/'observable-contexts.npz') as z:contexts=z['train'].copy()
    with np.load(hdir/'control-features.npz') as z:features=z['H128_PAD1024'].copy()
    y=[];x=[];cosines=[];groups={}
    for i,(row,record,label) in enumerate(zip(rows,records,labels['cases'])):
        if (row['index'],row['group_id'],row['style_index'])!=(record['index'],record['group_id'],record['style_index']) or row['index']!=label['index']:raise ValueError('Training mapping mismatch')
        groups.setdefault(row['group_id'],[]).append(i);pair=[]
        for role in ['content','reference']:
            if row[role]!=record['roles'][role]['original_path']:raise ValueError('Feature role mismatch')
            a=features[record['roles'][role]['feature_index']].astype(np.float64);pair.append(a/np.linalg.norm(a))
        a,b=pair;x.append(np.concatenate([a,b,b-a]));cosines.append(float(np.clip(1-a@b,0,2)));entry=label['files']['conditions'];p=labels_dir/entry['path']
        if digest(p)!=entry['sha256']:raise ValueError('Teacher changed')
        with np.load(p) as z:y.append(z['selected_delta'].copy())
    if len(groups)!=256 or contexts.shape!=(1024,128) or any(len(g)!=4 or sorted(rows[i]['style_index'] for i in g)!=list(range(4)) for g in groups.values()):raise ValueError('Bad groups')
    names=sorted(groups);rng=np.random.default_rng(20261127);names=[names[int(i)] for i in rng.permutation(256)];fold_groups=[names[i:i+64] for i in range(0,256,64)];x=np.stack(x);y=np.stack(y).astype(np.float64);predicted=np.empty((1024,128));risk=np.empty((1024,5));assignment=np.full(1024,-1,int)
    out=base/'condition-risk-oof-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();sources=['prepare_condition_risk_oof.py','condition-risk-oof-protocol.md','prepared_condition.py','train_triplet_pilot.py','check_petal_condition.py','training_checks.py']
    for n in sources:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Train-only four target-group folds for condition predictor; frozen backbone previously trained on this cohort',source_sha256={n:digest(frozen/n) for n in sources},cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),teacher_results_sha256=digest(labels_dir/'results.json'),teacher_audit_sha256=digest(labels_dir/'artifact-audit.json'),context_results_sha256=digest(context_dir/'results.json'),context_features_sha256=context_report['observable_contexts_sha256'],h128_features_sha256=hreport['control_features_sha256'],seed=20261127,folds=[],cases=[],development_GT_used=False,gate_trained=False,product_integrated=False,finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for fold,held_groups in enumerate(fold_groups):
        test=np.array([i for g in held_groups for i in groups[g]],int);fit=np.array([i for i in range(1024) if i not in set(test)],int);scale=float(np.sqrt(np.mean(np.sum(contexts[fit]**2,axis=1))));design=np.concatenate([x,contexts/scale],axis=1);mean=design[fit].mean(0);center=design[fit]-mean;ym=y[fit].mean(0);k=center@center.T;lam=.1*float(np.trace(k))/len(fit);dual=np.linalg.solve(k+lam*np.eye(len(fit)),y[fit]-ym);query=design[test]-mean;cross=query@center.T;pred=cross@dual+ym;predicted[test]=pred.clip(-3,3);assignment[test]=fold
        unexplained=np.maximum(0,np.sum(query**2,axis=1)-np.sum(cross*np.linalg.solve(k+lam*np.eye(len(fit)),cross.T).T,axis=1));risk[test]=np.stack([unexplained,np.linalg.norm(predicted[test],axis=1),np.abs(predicted[test]).max(1),np.linalg.norm(contexts[test],axis=1)/scale,np.asarray(cosines)[test]],axis=1)
        p=out/f'fold-{fold}.npz';np.savez_compressed(p,fit_indices=fit,test_indices=test,train_rms=np.array(scale),xmean=mean,center=center,ymean=ym,lambda_value=np.array(lam),dual=dual);report['folds'].append(dict(fold=fold,held_groups=held_groups,fit_count=len(fit),held_count=len(test),fit_sha256=digest(p)))
    if (assignment<0).any() or not np.isfinite(predicted).all() or not np.isfinite(risk).all():raise ValueError('Incomplete OOF predictions')
    p=out/'oof-conditions.npz';np.savez_compressed(p,predicted=predicted,fold=assignment,risk_features=risk);report['oof_conditions_sha256']=digest(p);report['risk_feature_names']=['unexplained_energy','condition_l2','condition_max_abs','context_norm_ratio','h128_cosine_distance']
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=labels['checkpoint_sha256']:raise ValueError('Wrong backbone')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    def read(name):
        entry=lookup[name];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Training image changed')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    with torch.no_grad():
        for i,row in enumerate(rows):
            c,s=[torch.from_numpy(read(row[k])[None]).permute(0,3,1,2).float().cuda()/255 for k in ['content','reference']];prepared=PreparedCondition(model,api,c,s);files={}
            for name,d in [('baseline',np.zeros(128,np.float32)),('oof',predicted[i].astype(np.float32))]:
                raw=prepared(torch.from_numpy(d[None]).cuda())[0][0].cpu().permute(1,2,0).numpy();rgb=np.rint(raw.clip(0,1)*255).astype(np.uint8);p=out/f'{row["index"]}-{name}.png';Image.fromarray(rgb).save(p);files[name]=dict(path=p.name,sha256=digest(p))
            if files['baseline']['sha256']!=labels['cases'][i]['files']['baseline']['sha256']:raise ValueError('Training baseline parity failed')
            report['cases'].append(dict(index=row['index'],group_id=row['group_id'],style_index=row['style_index'],fold=int(assignment[i]),files=files))
            if (i+1)%128==0:print(json.dumps(dict(predictions_saved=i+1,total=1024)),flush=True)
    p=out/'predictions.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(p);report['all_predictions_saved_before_target_GT_pixels']=True
    for row,case in zip(rows,report['cases']):
        gt=read(row['gt']);case['scores']={}
        for name,entry in case['files'].items():
            p=out/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('Prediction changed')
            rgb=np.asarray(Image.open(p).convert('RGB'));case['scores'][name]=float(np.abs(rgb.astype(float)-gt.astype(float)).mean())
    def summary(cases):
        a=np.array([v['scores']['baseline'] for v in cases]);b=np.array([v['scores']['oof'] for v in cases]);return dict(count=len(cases),baseline_mean=float(a.mean()),oof_mean=float(b.mean()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()))
    report['summary']=summary(report['cases']);report['fold_summaries']={str(f):summary([v for v in report['cases'] if v['fold']==f]) for f in range(4)};report['backbone_state_before']=before;report['backbone_state_after']=state_hash(model)
    if before!=report['backbone_state_after'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
