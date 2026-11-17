"""Cross-block oracle capacity; GT is intentionally used in fitting, never product."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from capacity_audit import block_split,native_features,inverse_post,post_forward,render_coefficients
from reference_transfer_probe import crossfit_transfer
from joint_guide import pooled_crossfit


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';prior=base/'reference-original-extension-v2';load=lambda p:json.loads(p.read_text());r=load(prior/'results.json');assert r['finished']
    data=base/'cohort-training-coverage-v2';assert digest(data/'cohort.json')==r['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==r['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']}
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    checkpoint=base/'multiref-control-v1/epoch-6.pth';assert digest(checkpoint)==r['checkpoint_sha256']
    out=base/'joint-guide-v1';out.mkdir(exist_ok=False);(out/'frozen-source').mkdir()
    names=['diagnose_joint_guide.py','joint_guide.py','joint-guide-protocol.md','test_joint_guide.py','capacity_audit.py','reference_transfer_probe.py']
    for n in names:(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='GT-assisted fixed-guide cross-block oracle on consumed train64; not inference or product',prior_sha256=digest(prior/'results.json'),checkpoint_sha256=digest(checkpoint),source_sha256={n:digest(out/'frozen-source'/n) for n in names},cases=[],finished=False,training=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    model,api,_=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    mask=block_split(256,256,32)
    def read(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def cross_render(rgb,coefficients):
        output=np.empty_like(rgb)
        for fold,train in enumerate([mask,~mask]):output.reshape(-1,3)[~train]=render_coefficients(model,api,rgb,coefficients[fold]).reshape(-1,3)[~train]
        return output
    for i,(row,previous) in enumerate(zip(r['selected'],r['cases'])):
        assert row['index']==previous['index'];index=row['index'];folder=out/str(index);folder.mkdir()
        original,reference,target,gt=[read(row[k]) for k in ['reference_original','reference','content','gt']]
        f=native_features(model,api,original);f.pop('guided_spatial');t=native_features(model,api,target);t.pop('guided_spatial')
        xf,xt=f['guided_global'],t['guided_global'];yf=inverse_post(reference,f['post_tables'])[0];yt=inverse_post(gt,t['post_tables'])[0]
        source=crossfit_transfer(xf,yf,xt,mask)
        old_coeff=prior/str(index)/'coefficients.npy';assert digest(old_coeff)==previous['coefficients_sha256']
        np.testing.assert_array_equal(source['coefficients'],np.load(old_coeff,allow_pickle=False))
        source_ref=cross_render(original,source['fold_coefficients']);source_target=render_coefficients(model,api,target,source['coefficients'])
        for label,a in [('reference-fit',source_ref),('transfer',source_target)]:
            e=previous['outputs'][label];p=prior/e['path'];assert digest(p)==e['sha256']
            with Image.open(p) as im:assert np.array_equal(a,np.array(im))
        own=crossfit_transfer(xt,yt,xt,mask);joint=pooled_crossfit(xf,yf,xt,yt,mask)
        outputs=dict(target_only=cross_render(target,own['fold_coefficients']),joint_target=cross_render(target,joint['fold_coefficients']),joint_reference=cross_render(original,joint['fold_coefficients']))
        expected=dict(target_only=post_forward(own['reference_crossfit'],t['post_tables']).reshape(target.shape),joint_target=post_forward(joint['b_crossfit'],t['post_tables']).reshape(target.shape),joint_reference=post_forward(joint['a_crossfit'],f['post_tables']).reshape(original.shape))
        maximum=max(int(np.abs(outputs[k].astype(int)-expected[k].astype(int)).max()) for k in outputs);assert maximum<=1
        severe=previous['transfer_mae']-previous['baseline_mae']>5 and previous['transfer_mae']>previous['baseline_mae']*1.25
        record=dict(index=index,previous_severe=severe,source_only_target_mae=previous['transfer_mae'],source_only_reference_mae=previous['reference_crossfit_mae'],baseline_mae=previous['baseline_mae'],native_max_code_difference=maximum,outputs={})
        for label,a in outputs.items():
            p=folder/(label+'.png');Image.fromarray(a).save(p)
            record['outputs'][label]=dict(sha256=digest(p),mae=float(np.abs(a.astype(float)-(reference if label=='joint_reference' else gt)).mean()))
        for label,coeff in [('target_only',own['fold_coefficients']),('joint',joint['fold_coefficients'])]:
            p=folder/(label+'-coefficients.npy');np.save(p,np.stack(coeff));record[label+'_coefficients_sha256']=digest(p)
        report['cases'].append(record);(out/'progress.json').write_text(json.dumps(report,indent=2))
        if (i+1)%8==0:print('completed',i+1,flush=True)
    summary={k:float(np.mean([c['outputs'][k]['mae'] for c in report['cases']])) for k in ['target_only','joint_target','joint_reference']}
    summary.update(source_only_target=float(np.mean([c['source_only_target_mae'] for c in report['cases']])),source_only_reference=float(np.mean([c['source_only_reference_mae'] for c in report['cases']])))
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(summary=summary,finished=True);(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(summary=summary,severe=[c for c in report['cases'] if c['previous_severe']])),flush=True)


if __name__=='__main__':main()
