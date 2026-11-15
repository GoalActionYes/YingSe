"""Fixed train-only logistic decision; selects saved PNGs, no GT at prediction."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from multiref_response import response_metrics


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';load=lambda p:json.loads(p.read_text(encoding='utf-8'));oof_dir=base/'condition-risk-oof-v1';oof=load(oof_dir/'results.json');audit=load(oof_dir/'artifact-audit.json');candidate_dir=base/'condition-context-ridge-v1';candidate=load(candidate_dir/'results.json');data=base/'cohort-multiref-v1';cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']};meta=load(base/'petal-condition-features-v1/results.json');records=[r for r in meta['rows'] if r['split']=='validation'];hreport=load(base/'petal-condition-training-v1/results.json')
    if not oof['finished'] or not audit['finished'] or audit['producer_results_sha256']!=digest(oof_dir/'results.json') or audit['pngs_scored']!=2048 or digest(data/'cohort.json')!=oof['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=oof['prepared_manifest_sha256']:raise ValueError('Training risk records not audited')
    if digest(oof_dir/'oof-conditions.npz')!=oof['oof_conditions_sha256'] or digest(candidate_dir/'predicted-conditions.npz')!=candidate['predicted_conditions_sha256']:raise ValueError('Conditions changed')
    with np.load(oof_dir/'oof-conditions.npz') as z:x=z['risk_features'].copy()
    y=np.array([r['scores']['oof']<r['scores']['baseline']-.1 for r in oof['cases']],float)
    if x.shape!=(1024,5) or y.shape!=(1024,) or {r['index'] for r in oof['cases']}&{r['index'] for r in cohort['groups']['validation']}:raise ValueError('Risk split leakage')
    mean=x.mean(0);std=np.maximum(x.std(0),1e-8);design=np.concatenate([np.ones((1024,1)),(x-mean)/std],1);weight=np.zeros(6);penalty=np.array([0.,1,1,1,1,1]);history=[]
    def objective(w):
        logits=np.clip(design@w,-40,40);return float(np.mean(np.logaddexp(0,logits)-y*logits)+.005*np.sum((w[1:])**2))
    for step in range(32):
        logits=np.clip(design@weight,-40,40);p=1/(1+np.exp(-logits));curvature=np.maximum(p*(1-p),1e-9);gradient=design.T@(p-y)/1024+.01*penalty*weight;hessian=(design.T*curvature)@design/1024+.01*np.diag(penalty);direction=np.linalg.solve(hessian,gradient);before=objective(weight);scale=0.
        for alpha in [1.,.5,.25,.125,.0625]:
            proposed=weight-alpha*direction
            if objective(proposed)<=before:weight=proposed;scale=alpha;break
        history.append(dict(step=step+1,objective=objective(weight),update_scale=scale,gradient_max_abs=float(np.abs(gradient).max())))
    out=base/'condition-risk-gate-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['probe_condition_risk_gate.py','condition-risk-gate-protocol.md','independent_coco.py','multiref_response.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Train-only fixed risk classifier, consumed64 development diagnostic, not calibrated probability',source_sha256={n:digest(frozen/n) for n in names},oof_results_sha256=digest(oof_dir/'results.json'),oof_audit_sha256=digest(oof_dir/'artifact-audit.json'),candidate_results_sha256=digest(candidate_dir/'results.json'),threshold=.5,l2=.01,history=history,training_positive_count=int(y.sum()),development_GT_used_for_fit=False,product_integrated=False,cases=[],finished=False)
    p=out/'classifier.npz';np.savez_compressed(p,weight=weight,train_mean=mean,train_std=std);report['classifier_sha256']=digest(p)
    with np.load(base/'petal-condition-training-v1/control-features.npz') as z:h=z['H128_PAD1024'].copy()
    if digest(base/'petal-condition-training-v1/control-features.npz')!=hreport['control_features_sha256']:raise ValueError('H128 changed')
    with np.load(candidate_dir/'observable-contexts.npz') as z:context=z['validation'].copy();context_rms=float(z['train_rms'])
    with np.load(candidate_dir/'predicted-conditions.npz') as z:deltas=z['HC-context'].copy()
    with np.load(candidate_dir/'HC-context-fit.npz') as z:
        vectors=[];distances=[]
        for record in records:
            a,b=[h[record['roles'][k]['feature_index']].astype(float) for k in ['content','reference']];a/=np.linalg.norm(a);b/=np.linalg.norm(b);vectors.append(np.concatenate([a,b,b-a]));distances.append(float(np.clip(1-a@b,0,2)))
        query=np.concatenate([np.stack(vectors),context/context_rms],1)-z['xmean'];cross=query@z['center'].T;k=z['center']@z['center'].T;unexplained=np.maximum(0,np.sum(query**2,1)-np.sum(cross*np.linalg.solve(k+float(z['lambda_value'])*np.eye(1024),cross.T).T,1))
    inputs=np.stack([unexplained,np.linalg.norm(deltas,axis=1),np.abs(deltas).max(1),np.linalg.norm(context,axis=1)/context_rms,np.array(distances)],1);test=np.concatenate([np.ones((64,1)),(inputs-mean)/std],1);scores=1/(1+np.exp(-np.clip(test@weight,-40,40)))
    if not np.isfinite(scores).all():raise ValueError('Nonfinite risk scores')
    p=out/'risk-inputs.npz';np.savez_compressed(p,risk_features=inputs,scores=scores);report['risk_inputs_sha256']=digest(p)
    for i,(row,case) in enumerate(zip(cohort['groups']['validation'],candidate['cases'])):
        if row['index']!=case['index'] or row['index']!=records[i]['index']:raise ValueError('Candidate mapping changed')
        choice='HC-context' if scores[i]>=.5 else 'baseline';entry=case['files'][choice];p=candidate_dir/entry['path']
        if digest(p)!=entry['sha256']:raise ValueError('Selected output changed')
        dst=out/f'{row["index"]}-selected.png';dst.write_bytes(p.read_bytes());report['cases'].append(dict(index=row['index'],group_id=row['group_id'],style_index=row['style_index'],score=float(scores[i]),choice=choice,file=dict(path=dst.name,sha256=digest(dst))))
    p=out/'predictions.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(p);report['all_choices_saved_before_development_GT_pixels']=True;groups={}
    for row,case,old in zip(cohort['groups']['validation'],report['cases'],candidate['cases']):
        entry=lookup[row['gt']];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('GT changed')
        with Image.open(p) as im:gt=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        rgb=np.asarray(Image.open(out/case['file']['path']).convert('RGB'));mae=float(np.abs(rgb.astype(float)-gt.astype(float)).mean());case['mae']=mae;case['baseline_mae']=old['scores']['baseline'];case['H128_mae']=old['scores']['H128'];case['raw_candidate_mae']=old['scores']['HC-context'];groups.setdefault(row['group_id'],{})[row['style_index']]=(rgb,gt)
        if mae!=old['scores'][case['choice']]:raise ValueError('Selection score mismatch')
    report['paired']=[dict(group_id=k,**response_metrics(v[0][0],v[1][0],v[0][1],v[1][1])) for k,v in groups.items()];a=np.array([r['baseline_mae'] for r in report['cases']]);b=np.array([r['mae'] for r in report['cases']]);c=np.array([r['H128_mae'] for r in report['cases']]);raw=np.array([r['raw_candidate_mae'] for r in report['cases']]);accepted=np.array([r['choice']=='HC-context' for r in report['cases']]);severe=(b>a+5)&(b>a*1.25)
    report['summary']=dict(baseline_mean=float(a.mean()),gated_mean=float(b.mean()),raw_candidate_mean=float(raw.mean()),accepted=int(accepted.sum()),fallback=int((~accepted).sum()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()),severe=int(severe.sum()),beneficial_raw_rejected=int(((raw<a-.1)&(~accepted)).sum()),severe_raw_retained=int((((raw>a+5)&(raw>a*1.25))&accepted).sum()))
    def compare(before,ratio,count):return bool(b.mean()<=before.mean()*(1-ratio) and (b<before-.1).sum()>=count and ((b>before+5)&(b>before*1.25)).sum()<=2)
    select=np.array([r['style_index']==1 for r in report['cases']]);pa=np.array([r['delta_mae'] for r in candidate['paired']['H128']]);pb=np.array([r['delta_mae'] for r in report['paired']]);gates=dict(vs_current=compare(a,.05,40),vs_H128=compare(c,.03,36),added=bool(b[select].mean()<=a[select].mean() and (b[select]<a[select]-.1).sum()>=17),response=bool(pb.mean()<=.95*pa.mean() and (pb<pa-.1).sum()>=20));gates['passed']=all(gates.values());report['gates']=gates
    if any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()) or digest(oof_dir/'results.json')!=report['oof_results_sha256'] or digest(candidate_dir/'results.json')!=report['candidate_results_sha256']:raise ValueError('Frozen inputs/source changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=report['summary'],gates=gates)),flush=True)


if __name__=='__main__':main()
