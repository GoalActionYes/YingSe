"""Bounded reference-original diagnosis; all samples are consumed training data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output
from capacity_audit import block_split,polynomial_features,native_features,inverse_post,post_forward,render_coefficients
from reference_transfer_probe import crossfit_transfer

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='reference-transfer-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected output directory name')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';index=ROOT/'.tools/TST100K-index/triplet.json';cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';checkpoint=base/'multiref-control-v1/epoch-6.pth'
    expected={index:'5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413',cohort_path:'124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d',prep_path:'b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c',checkpoint:START_SHA}
    if any(digest(p)!=sha for p,sha in expected.items()):raise ValueError('Changed probe inputs')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort,prep,rows=load(cohort_path),load(prep_path),load(index)
    if not prep['passed'] or prep['cohort_sha256']!=expected[cohort_path] or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Unexpected prepared cohort')
    lookup={r['original_path']:r for r in prep['files']};originals=defaultdict(set)
    for row in rows:originals[Path(row['content']).stem].add(row['content'])
    selected=[]
    for row in cohort['groups']['train']:
        options=originals[row['reference_family']]
        if len(options)!=1:raise ValueError('Ambiguous original reference')
        original=next(iter(options))
        if original in lookup:selected.append(dict(row,reference_original=original))
    if len(selected)!=75 or len({r['reference_original'] for r in selected})!=47:raise ValueError('Changed eligibility; do not select a different cohort')
    out=base/args.run;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['run_reference_transfer.py','reference_transfer_probe.py','reference-transfer-protocol.md','capacity_audit.py','train_triplet_pilot.py','training_checks.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Original-reference extra-information diagnosis, consumed training75 only; not a two-image inference algorithm or independent acceptance',config=dict(size=256,tile=32,ridge=1e-5,coefficient_combination='mean of two checkerboard fold matrices'),checkpoint_sha256=START_SHA,cohort_sha256=expected[cohort_path],prepared_manifest_sha256=expected[prep_path],index_sha256=expected[index],source_sha256={p.name:digest(p) for p in frozen.iterdir()},selected=selected,cases=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,manifest=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite current weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);report['author_revision']=manifest['revision']
    mask=block_split(256,256,32);started=time.perf_counter()
    def read(relative):
        entry=lookup[relative];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Changed prepared image')
        with Image.open(p) as im:
            size=im.size;rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return rgb,list(size)
    def tensor(rgb):return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    def errors(pred,gt):
        delta=np.abs(pred.astype(float)-gt);return dict(mae=float(delta.mean()),p95=float(np.quantile(delta,.95)))
    def quantize(raw):return np.rint(np.clip(raw,0,1)*255).astype(np.uint8).reshape(256,256,3)
    def save(label,rgb,index):
        p=out/f'{index}-{label}.png';Image.fromarray(rgb).save(p);return digest(p)
    def key(p):return Path(p).stem.rsplit('_',1)[-1]
    for number,row in enumerate(selected):
        content,cs=read(row['content']);reference,rs=read(row['reference']);truth,ts=read(row['gt']);original,os=read(row['reference_original'])
        with torch.inference_mode():
            c,s=tensor(content),tensor(reference);hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));raw,_=model(c,hc,hs)
            finite_output(raw,c.shape);current=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
        case=dict(index=row['index'],filename_key_equal=key(row['reference'])==key(row['gt']),prepared_sizes=dict(content=cs,reference=rs,gt=ts,reference_original=os),reference_original_style_sizes_equal=os==rs,current=errors(current,truth),methods={},sha256={'current':save('current',current,row['index'])},
            target_channel_outside_reference_minmax_fraction=np.mean((content.astype(float)<original.reshape(-1,3).min(0))|(content.astype(float)>original.reshape(-1,3).max(0)),axis=(0,1)).tolist())
        target_features=native_features(model,api,content);target_features.pop('guided_spatial')
        source_features=native_features(model,api,original);source_features.pop('guided_spatial')
        transfer_outputs={};reference_outputs={};oracle_outputs={}
        for method in ['quadratic','guided_global']:
            guided=method=='guided_global'
            source=source_features[method] if guided else polynomial_features(original,2)
            target=target_features[method] if guided else polynomial_features(content,2)
            ref_values=inverse_post(reference,source_features['post_tables'])[0] if guided else reference.reshape(-1,3).astype(float)/255
            gt_values=inverse_post(truth,target_features['post_tables'])[0] if guided else truth.reshape(-1,3).astype(float)/255
            fitted=crossfit_transfer(source,ref_values,target,mask);self_fit=crossfit_transfer(target,gt_values,target,mask)
            if guided:
                render=lambda values,tables:post_forward(values,tables).reshape(content.shape)
                ref_fit=render(fitted['reference_crossfit'],source_features['post_tables']);transfer=render(fitted['target_prediction'],target_features['post_tables']);own=render(self_fit['reference_crossfit'],target_features['post_tables'])
                native_ref=np.empty_like(original);native_own=np.empty_like(content)
                for fold,train in enumerate([mask,~mask]):
                    native_ref.reshape(-1,3)[~train]=render_coefficients(model,api,original,fitted['fold_coefficients'][fold]).reshape(-1,3)[~train]
                    native_own.reshape(-1,3)[~train]=render_coefficients(model,api,content,self_fit['fold_coefficients'][fold]).reshape(-1,3)[~train]
                native_transfer=render_coefficients(model,api,content,fitted['coefficients'])
                maximum=max(int(np.abs(a.astype(int)-b.astype(int)).max()) for a,b in [(ref_fit,native_ref),(transfer,native_transfer),(own,native_own)])
                if maximum>1:raise ValueError('Dense/native probe disagreement')
                ref_fit,transfer,own=native_ref,native_transfer,native_own
            else:
                ref_fit=quantize(fitted['reference_crossfit']);transfer=quantize(fitted['target_prediction']);own=quantize(self_fit['reference_crossfit']);maximum=None
            case['methods'][method]=dict(reference_crossfit=errors(ref_fit,reference),target_reference_transfer=errors(transfer,truth),target_gt_crossfit=errors(own,truth),native_max_code_difference=maximum,maximum_reference_coefficient=float(np.abs(fitted['coefficients']).max()))
            for name,im in [('reference-fit',ref_fit),('target-transfer',transfer),('target-gt-fit',own)]:case['sha256'][method+'-'+name]=save(method+'-'+name,im,row['index'])
            transfer_outputs[method]=transfer;reference_outputs[method]=ref_fit;oracle_outputs[method]=own
        canvas=Image.new('RGB',(1024,825),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',content),('Styled reference',reference),('Target GT',truth),('Current13 (trained subset)',current),('Original reference (extra info)',original),('Ref cross-fit quadratic',reference_outputs['quadratic']),('Ref cross-fit guided',reference_outputs['guided_global']),('Target transfer quadratic',transfer_outputs['quadratic']),('Target transfer guided',transfer_outputs['guided_global']),('Target GT self-fit quadratic',oracle_outputs['quadratic']),('Target GT self-fit guided',oracle_outputs['guided_global'])]
        for i,(label,im) in enumerate(panels):
            x=i%4*256;y=i//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(im),(x,y+19))
        canvas.save(out/f'{row["index"]}.jpg',quality=95);report['cases'].append(case)
        del source_features,target_features
        (out/'progress.json').write_text(json.dumps(dict(completed=number+1,total=75)),encoding='utf-8')
        if (number+1)%10==0 or number==0:print(json.dumps(dict(completed=number+1,total=75)),flush=True)
    report['summary']={}
    for subset in ['all','filename_key_equal','filename_key_unequal']:
        cases=[r for r in report['cases'] if subset=='all' or r['filename_key_equal']==(subset=='filename_key_equal')]
        if not cases:raise ValueError('Empty diagnostic stratum')
        summary=dict(cases=len(cases),current_mean=float(np.mean([r['current']['mae'] for r in cases])),methods={})
        for method in ['quadratic','guided_global']:
            vals=[r['methods'][method] for r in cases];summary['methods'][method]={kind:dict(mean=float(np.mean([r[kind]['mae'] for r in vals])),median=float(np.median([r[kind]['mae'] for r in vals])),at_most_3=sum(r[kind]['mae']<=3 for r in vals),at_most_5=sum(r[kind]['mae']<=5 for r in vals)) for kind in ['reference_crossfit','target_reference_transfer','target_gt_crossfit']}
            summary['methods'][method]['improved_vs_current']=sum(r['current']['mae']-r['methods'][method]['target_reference_transfer']['mae']>.1 for r in cases)
        report['summary'][subset]=summary
    if len(report['cases'])!=75 or any(digest(p)!=sha for p,sha in expected.items()):raise ValueError('Incomplete or changed evidence')
    report.update(finished=True,wall_seconds=time.perf_counter()-started)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,summary=report['summary'],wall_seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
