"""CPU fold/linear-system and saved-image audit, not a risk calibrator."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'condition-risk-oof-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));report=load(out/'results.json');data=base/'cohort-multiref-v1';cohort=load(data/'cohort.json');rows=cohort['groups']['train'];lookup={v['original_path']:v for v in load(data/'srgb-v1/manifest.json')['files']};meta=load(base/'petal-condition-features-v1/results.json');records=[r for r in meta['rows'] if r['split']=='train'];ldir=base/'condition-labels-train-v1';labels=load(ldir/'results.json')
    if not report['finished'] or report['development_GT_used'] or len(report['cases'])!=1024 or digest(data/'cohort.json')!=report['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=report['prepared_manifest_sha256'] or digest(ldir/'results.json')!=report['teacher_results_sha256']:raise ValueError('Wrong data/scope')
    for n,sha in report['source_sha256'].items():
        if digest(Path(__file__).with_name(n))!=sha or digest(out/'frozen-source'/n)!=sha:raise ValueError('Frozen source changed')
    with np.load(base/'condition-context-ridge-v1/observable-contexts.npz') as z:ctx=z['train'].copy()
    with np.load(base/'petal-condition-training-v1/control-features.npz') as z:features=z['H128_PAD1024'].copy()
    x=[];y=[];cos=[];groups={}
    for i,(row,record,label) in enumerate(zip(rows,records,labels['cases'])):
        if row['index']!=record['index'] or row['index']!=label['index']:raise ValueError('Input mapping changed')
        groups.setdefault(row['group_id'],[]).append(i);a,b=[features[record['roles'][k]['feature_index']].astype(float) for k in ['content','reference']];a/=np.linalg.norm(a);b/=np.linalg.norm(b);x.append(np.concatenate([a,b,b-a]));cos.append(float(np.clip(1-a@b,0,2)))
        p=ldir/label['files']['conditions']['path']
        if digest(p)!=label['files']['conditions']['sha256']:raise ValueError('Teacher changed')
        with np.load(p) as z:y.append(z['selected_delta'].copy())
    x=np.stack(x);y=np.stack(y).astype(float);names=sorted(groups);rng=np.random.default_rng(20261127);names=[names[int(i)] for i in rng.permutation(256)]
    if digest(out/'oof-conditions.npz')!=report['oof_conditions_sha256']:raise ValueError('OOF predictions changed')
    with np.load(out/'oof-conditions.npz') as z:pred=z['predicted'].copy();risk=z['risk_features'].copy();fold=z['fold'].copy()
    maximum_residual=0.
    for f in range(4):
        held=names[f*64:(f+1)*64];test=np.array([i for g in held for i in groups[g]],int);fit=np.array([i for i in range(1024) if i not in set(test)],int);p=out/f'fold-{f}.npz'
        if digest(p)!=report['folds'][f]['fit_sha256'] or held!=report['folds'][f]['held_groups']:raise ValueError('Fold changed')
        with np.load(p) as z:
            if not np.array_equal(z['fit_indices'],fit) or not np.array_equal(z['test_indices'],test) or not np.all(fold[test]==f):raise ValueError('Group leakage')
            scale=float(np.sqrt(np.mean(np.sum(ctx[fit]**2,axis=1))));design=np.concatenate([x,ctx/scale],axis=1);mean=design[fit].mean(0);center=design[fit]-mean;ym=y[fit].mean(0);k=center@center.T;lam=.1*float(np.trace(k))/768;query=design[test]-mean;cross=query@center.T
            for name,value in [('train_rms',scale),('xmean',mean),('center',center),('ymean',ym),('lambda_value',lam)]:
                if not np.allclose(z[name],value,rtol=1e-12,atol=1e-12):raise ValueError('Fold-only scaling mismatch')
            residual=float(np.abs((k+lam*np.eye(768))@z['dual']-(y[fit]-ym)).max());maximum_residual=max(maximum_residual,residual)
            expected=(cross@z['dual']+ym).clip(-3,3);unexplained=np.maximum(0,np.sum(query**2,axis=1)-np.sum(cross*np.linalg.solve(k+lam*np.eye(768),cross.T).T,axis=1));rr=np.stack([unexplained,np.linalg.norm(expected,axis=1),np.abs(expected).max(1),np.linalg.norm(ctx[test],axis=1)/scale,np.asarray(cos)[test]],axis=1)
            if residual>1e-9 or not np.allclose(expected,pred[test],rtol=1e-10,atol=1e-10) or not np.allclose(rr,risk[test],rtol=1e-9,atol=1e-10):raise ValueError('OOF solve/risk mismatch')
    measured=[]
    for i,(row,case) in enumerate(zip(rows,report['cases'])):
        if row['index']!=case['index'] or case['fold']!=fold[i]:raise ValueError('Render order changed')
        p=data/lookup[row['gt']]['prepared_path']
        if digest(p)!=lookup[row['gt']]['sha256']:raise ValueError('GT changed')
        with Image.open(p) as im:gt=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        scores={}
        for name,entry in case['files'].items():
            p=out/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('PNG changed')
            a=np.asarray(Image.open(p).convert('RGB'));scores[name]=float(np.abs(a.astype(float)-gt.astype(float)).mean())
        if scores!=case['scores'] or case['files']['baseline']['sha256']!=labels['cases'][i]['files']['baseline']['sha256']:raise ValueError('Score/base mismatch')
        measured.append(dict(fold=case['fold'],scores=scores))
    def summary(cases):
        a=np.array([r['scores']['baseline'] for r in cases]);b=np.array([r['scores']['oof'] for r in cases]);return dict(count=len(cases),baseline_mean=float(a.mean()),oof_mean=float(b.mean()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()))
    if summary(measured)!=report['summary'] or {str(f):summary([r for r in measured if r['fold']==f]) for f in range(4)}!=report['fold_summaries'] or report['backbone_state_before']!=report['backbone_state_after']:raise ValueError('Summary/state mismatch')
    result=dict(scope='CPU train-only grouped OOF artifact audit, no gate validation or GPU replay',producer_results_sha256=digest(out/'results.json'),auditor_sha256=digest(Path(__file__)),folds_checked=4,condition_rows_checked=1024,pngs_scored=2048,maximum_linear_system_residual=maximum_residual,summary=report['summary'],gate_trained=False,finished=True);p=out/'artifact-audit.json'
    if p.exists():raise ValueError('Refusing overwrite')
    p.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
