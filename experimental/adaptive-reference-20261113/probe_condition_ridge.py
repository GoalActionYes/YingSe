"""Fixed train-only centered ridge; predictions frozen before development GT read."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash
from multiref_response import response_metrics


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']};cache=base/'petal-condition-features-v1';meta=load(cache/'results.json');teacher_dir=base/'condition-labels-train-v1';teachers=load(teacher_dir/'results.json');audit=load(teacher_dir/'artifact-audit.json');control_dir=base/'petal-condition-training-v1';control_report=load(control_dir/'results.json')
    if not teachers['finished'] or audit['producer_results_sha256']!=digest(teacher_dir/'results.json') or audit['condition_snapshots_checked']!=9216 or digest(data/'cohort.json')!=meta['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=meta['prepared_manifest_sha256']:raise ValueError('Data provenance mismatch')
    if digest(cache/'features.npz')!=meta['features_sha256'] or digest(control_dir/'control-features.npz')!=control_report['control_features_sha256']:raise ValueError('Feature cache changed')
    with np.load(cache/'features.npz') as z:petal=z['PETAL1024'].copy()
    with np.load(control_dir/'control-features.npz') as z:h128=z['H128_PAD1024'].copy()
    records={s:[r for r in meta['rows'] if r['split']==s] for s in ['train','validation']};ys=[]
    for row,record,teacher in zip(cohort['groups']['train'],records['train'],teachers['cases']):
        if row['index']!=record['index'] or row['index']!=teacher['index']:raise ValueError('Teacher ordering mismatch')
        entry=teacher['files']['conditions'];p=teacher_dir/entry['path']
        if digest(p)!=entry['sha256']:raise ValueError('Teacher file changed')
        with np.load(p) as z:ys.append(z['selected_delta'].copy())
    y=np.stack(ys).astype(np.float64)
    if y.shape!=(1024,128) or len(records['validation'])!=64 or {r['index'] for r in records['validation']}&{r['index'] for r in records['train']}:raise ValueError('Wrong split')
    def design(features,split):
        vectors=[]
        for row,record in zip(cohort['groups'][split],records[split]):
            a,b=[features[record['roles'][role]['feature_index']].astype(np.float64) for role in ['content','reference']]
            if any(row[role]!=record['roles'][role]['original_path'] for role in ['content','reference']):raise ValueError('Feature role mismatch')
            a/=np.linalg.norm(a);b/=np.linalg.norm(b);vectors.append(np.concatenate([a,b,b-a]))
        return np.stack(vectors)
    out=base/'condition-ridge-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['probe_condition_ridge.py','condition-ridge-protocol.md','prepared_condition.py','train_triplet_pilot.py','check_petal_condition.py','training_checks.py','multiref_response.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Consumed1024train/64development fixed ridge, not independent acceptance',source_sha256={n:digest(frozen/n) for n in names},teacher_results_sha256=digest(teacher_dir/'results.json'),teacher_audit_sha256=digest(teacher_dir/'artifact-audit.json'),feature_report_sha256=digest(cache/'results.json'),features_sha256=dict(PETAL1024=meta['features_sha256'],H128=control_report['control_features_sha256']),development_GT_used_for_training=False,product_integrated=False,models={},cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');predictions={'mean':np.repeat(y.mean(0)[None],64,0)};ymean=y.mean(0);fits={}
    for name,features in [('H128',h128),('PETAL1024',petal)]:
        start=time.perf_counter();x=design(features,'train');test=design(features,'validation');mean=x.mean(0);center=x-mean;k=center@center.T;lam=.1*float(np.trace(k))/1024
        if not lam>0 or not np.isfinite(x).all():raise ValueError('Degenerate design')
        dual=np.linalg.solve(k+lam*np.eye(1024),y-ymean);pred=(test-mean)@center.T@dual+ymean;train=center@center.T@dual+ymean
        if not np.isfinite(pred).all():raise ValueError('Nonfinite ridge output')
        predictions[name]=pred.clip(-3,3);fits[name]=dict(xmean=mean,center=center,dual=dual,lambda_value=np.array(lam),ymean=ymean)
        p=out/f'{name}-fit.npz';np.savez_compressed(p,**fits[name]);report['models'][name]=dict(lambda_value=lam,train_label_mse=float(((train-y)**2).mean()),unclipped_prediction_max_abs=float(np.abs(pred).max()),clipped_components=int((np.abs(pred)>3).sum()),seconds=time.perf_counter()-start,fit_sha256=digest(p))
    predictions['mean']=predictions['mean'].clip(-3,3);p=out/'predicted-conditions.npz';np.savez_compressed(p,**predictions);report['predicted_conditions_sha256']=digest(p)
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=teachers['checkpoint_sha256']:raise ValueError('Wrong backbone')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    def image(name):
        entry=lookup[name];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Image changed')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    original=load(control_dir/'PETAL1024/epoch-0/scores.json')
    for i,row in enumerate(cohort['groups']['validation']):
        c,s=[torch.from_numpy(image(row[k])[None]).permute(0,3,1,2).float().cuda()/255 for k in ['content','reference']];prepared=PreparedCondition(model,api,c,s);files={}
        with torch.no_grad():
            for name in ['baseline','mean','H128','PETAL1024']:
                d=np.zeros(128,np.float32) if name=='baseline' else predictions[name][i].astype(np.float32);raw=prepared(torch.from_numpy(d[None]).cuda())[0][0].cpu().permute(1,2,0).numpy();rgb=np.rint(raw.clip(0,1)*255).astype(np.uint8);p=out/f'{row["index"]}-{name}.png';Image.fromarray(rgb).save(p);files[name]=dict(path=p.name,sha256=digest(p))
        if files['baseline']['sha256']!=original['cases'][i]['sha256']:raise ValueError('Historical baseline parity failed')
        report['cases'].append(dict(index=row['index'],group_id=row['group_id'],style_index=row['style_index'],files=files))
    p=out/'predictions.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(p);report['all_predictions_saved_before_GT_pixels']=True;groups={}
    for row,case in zip(cohort['groups']['validation'],report['cases']):
        truth=image(row['gt']);case['scores']={};g=groups.setdefault(row['group_id'],{})
        for name,entry in case['files'].items():
            p=out/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('Output changed')
            rgb=np.asarray(Image.open(p).convert('RGB'));case['scores'][name]=float(np.abs(rgb.astype(float)-truth.astype(float)).mean());g.setdefault(name,{})[row['style_index']]=(rgb,truth)
    report['paired']={name:[dict(group_id=key,**response_metrics(v[name][0][0],v[name][1][0],v[name][0][1],v[name][1][1])) for key,v in groups.items()] for name in ['baseline','mean','H128','PETAL1024']}
    report['summary']={name:dict(mean=float(np.mean([c['scores'][name] for c in report['cases']])),delta_mean=float(np.mean([r['delta_mae'] for r in report['paired'][name]]))) for name in report['paired']}
    def compare(before,ratio,count):
        a=np.array([r['scores'][before] for r in report['cases']]);b=np.array([r['scores']['PETAL1024'] for r in report['cases']]);return dict(improved=int((b<a-.1).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()),passed=bool(b.mean()<=a.mean()*(1-ratio) and (b<a-.1).sum()>=count and ((b>a+5)&(b>a*1.25)).sum()<=2))
    gates=dict(vs_current=compare('baseline',.05,40),vs_matched=compare('H128',.03,36));added=[r for r in report['cases'] if r['style_index']==1];a=np.array([r['scores']['baseline'] for r in added]);b=np.array([r['scores']['PETAL1024'] for r in added]);gates['added_passed']=bool(b.mean()<=a.mean() and (b<a-.1).sum()>=17);a=np.array([r['delta_mae'] for r in report['paired']['H128']]);b=np.array([r['delta_mae'] for r in report['paired']['PETAL1024']]);gates['response_passed']=bool(b.mean()<=.95*a.mean() and (b<a-.1).sum()>=20);gates['passed']=bool(gates['vs_current']['passed'] and gates['vs_matched']['passed'] and gates['added_passed'] and gates['response_passed']);report['gates']=gates
    report['backbone_state_before']=before;report['backbone_state_after']=state_hash(model)
    if before!=report['backbone_state_after'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen source/model changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=report['summary'],gates=gates)),flush=True)


if __name__=='__main__':main()
