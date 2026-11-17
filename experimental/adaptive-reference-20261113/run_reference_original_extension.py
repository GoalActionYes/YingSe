"""Additional-reference information diagnosis; target GT absent from prediction."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from capacity_audit import block_split,native_features,inverse_post,post_forward,render_coefficients
from reference_transfer_probe import crossfit_transfer


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';coverage=base/'reference-original-coverage-v1'
    load=lambda p:json.loads(p.read_text())
    audit=load(coverage/'results.json');old_meta=load(coverage/'cohort-multiref-v1.json');new_meta=load(coverage/'cohort-training-coverage-v2.json')
    for name in ['cohort-multiref-v1','cohort-training-coverage-v2']:
        assert digest(coverage/(name+'.json'))==audit['cohorts'][name]['detailed_training_report_sha256']
    data=base/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json')
    assert digest(data/'cohort.json')==audit['cohorts'][data.name]['cohort_sha256']
    assert digest(data/'srgb-v1/manifest.json')==audit['cohorts'][data.name]['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in prep['files']};rows={r['index']:r for r in cohort['groups']['train']}
    old_indices={r[0] for r in old_meta['selected']};seen=set();selected=[]
    for index,original,reference in new_meta['selected']:
        row=rows[index]
        if index in old_indices or row['content'] in seen:continue
        assert row['reference']==reference
        selected.append(dict(row,reference_original=original));seen.add(row['content'])
        if len(selected)==64:break
    assert len(selected)==len(seen)==64
    prior=base/'reference-transfer-v1';history=load(prior/'results.json');assert history['finished']
    dependencies=['capacity_audit.py','reference_transfer_probe.py','train_triplet_pilot.py','training_checks.py']
    for n in dependencies:assert digest(Path(__file__).with_name(n))==history['source_sha256'][n]
    checkpoint=base/'multiref-control-v1/epoch-6.pth';assert digest(checkpoint)==history['checkpoint_sha256']
    out=base/'reference-original-extension-v2';out.mkdir(exist_ok=False);(out/'frozen-source').mkdir()
    names=dependencies+['run_reference_original_extension.py','reference-original-extension-protocol.md']
    for n in names:(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Consumed train64, extra reference original supplied; not two-image product inference',
        checkpoint_sha256=digest(checkpoint),coverage_report_sha256=digest(coverage/'results.json'),
        cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),
        prior_results_sha256=digest(prior/'results.json'),selected=selected,
        source_sha256={n:digest(out/'frozen-source'/n) for n in names},cases=[],finished=False,training=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    model,api,_=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    mask=block_split(256,256,32)
    def read(data,lookup,p):
        e=lookup[p];path=data/e['prepared_path'];assert digest(path)==e['sha256']
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)),im.size
    def predict(row,data,lookup,folder):
        folder.mkdir();content,_=read(data,lookup,row['content']);original,osize=read(data,lookup,row['reference_original']);reference,rsize=read(data,lookup,row['reference'])
        assert osize==rsize
        with torch.inference_mode():
            c,s=[(torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255).cuda() for a in [content,reference]]
            hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
            raw,_=model(c,hc,hs);assert torch.isfinite(raw).all();baseline=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
        f=native_features(model,api,original);f.pop('guided_spatial');t=native_features(model,api,content);t.pop('guided_spatial')
        values=inverse_post(reference,f['post_tables'])[0]
        fit=crossfit_transfer(f['guided_global'],values,t['guided_global'],mask)
        dense_ref=post_forward(fit['reference_crossfit'],f['post_tables']).reshape(reference.shape)
        dense_target=post_forward(fit['target_prediction'],t['post_tables']).reshape(content.shape)
        native_ref=np.empty_like(original)
        for fold,train in enumerate([mask,~mask]):native_ref.reshape(-1,3)[~train]=render_coefficients(model,api,original,fit['fold_coefficients'][fold]).reshape(-1,3)[~train]
        transfer=render_coefficients(model,api,content,fit['coefficients'])
        maximum=max(int(np.abs(a.astype(int)-b.astype(int)).max()) for a,b in [(dense_ref,native_ref),(dense_target,transfer)])
        assert maximum<=1
        record=dict(index=row['index'],native_max_code_difference=maximum,
            reference_crossfit_mae=float(np.abs(native_ref.astype(float)-reference).mean()),outputs={})
        for label,a in [('baseline',baseline),('transfer',transfer),('reference-fit',native_ref)]:
            p=folder/(label+'.png');Image.fromarray(a).save(p);record['outputs'][label]=dict(path=str(p.relative_to(out)),sha256=digest(p))
        p=folder/'coefficients.npy';np.save(p,fit['coefficients']);record['coefficients_sha256']=digest(p)
        return record
    old_data=base/'cohort-multiref-v1';old_lookup={e['original_path']:e for e in load(old_data/'srgb-v1/manifest.json')['files']}
    bridge_row=history['selected'][0];bridge=predict(bridge_row,old_data,old_lookup,out/'bridge');saved=history['cases'][0];assert bridge['index']==saved['index']
    for label,old_label in [('baseline','current'),('transfer','guided_global-target-transfer'),('reference-fit','guided_global-reference-fit')]:
        p=prior/f"{bridge['index']}-{old_label}.png";assert digest(p)==saved['sha256'][old_label]
        with Image.open(p) as im:old_pixels=np.array(im)
        with Image.open(out/bridge['outputs'][label]['path']) as im:actual=np.array(im)
        if not np.array_equal(old_pixels,actual):
            delta=np.abs(old_pixels.astype(int)-actual.astype(int))
            (out/'bridge-failure.json').write_text(json.dumps(dict(label=label,maximum=int(delta.max()),unequal_channels=int((delta>0).sum()))))
            raise ValueError('Historical RGB bridge differs; see bridge-failure.json')
    report['bridge']=dict(index=bridge['index'],three_outputs_pixel_exact=True)
    for i,row in enumerate(selected):
        r=predict(row,data,lookup,out/str(row['index']));report['cases'].append(r)
        (out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(predicted=i+1,total=64)),flush=True)
    # No target GT has been read before this point.
    for row,r in zip(selected,report['cases']):
        gt,_=read(data,lookup,row['gt'])
        for label in ['baseline','transfer']:
            e=r['outputs'][label];assert digest(out/e['path'])==e['sha256']
            with Image.open(out/e['path']) as im:a=np.array(im)
            r[label+'_mae']=float(np.abs(a.astype(float)-gt).mean())
    b=np.array([r['baseline_mae'] for r in report['cases']]);c=np.array([r['transfer_mae'] for r in report['cases']])
    summary=dict(baseline_mean=float(b.mean()),transfer_mean=float(c.mean()),relative_improvement=float(1-c.mean()/b.mean()),improved=int((b-c>.1).sum()),severe=int(((c-b>5)&(c>b*1.25)).sum()),reference_crossfit_mean=float(np.mean([r['reference_crossfit_mae'] for r in report['cases']])))
    summary['supervision_probe_passed']=bool(summary['relative_improvement']>=.05 and summary['improved']>=40 and summary['severe']<=2)
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(summary=summary,finished=True);(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
