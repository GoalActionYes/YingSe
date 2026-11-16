"""Post-failure fixed checkpoint interventions; no candidate selection or training."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from regional_correction import RegionalCorrection,global_control
from spatial_render import SpatialEngine


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';cache=b/'regional-correction-inputs-v1';train=b/'regional-correction-training-v1'
    info=json.loads((cache/'results.json').read_text());run=json.loads((train/'results.json').read_text());assert run['finished'] and run['arms']['regional']['best_epoch']==3
    for n,h in run['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    assert digest(cache/'results.json')==run['input_report_sha256'] and digest(cache/'descriptors.npz')==info['descriptors_sha256']
    with np.load(cache/'descriptors.npz') as f:features=f['regional']
    inputs={i['index']:i for i in info['inputs']};rows=[r for r in info['rows'] if r['split']=='validation']
    cohort=json.loads((b/'cohort-multiref-v1/cohort.json').read_text());assert [r['index'] for r in rows]==[r['index'] for r in cohort['groups']['validation']]
    out=b/'regional-correction-diagnostic-v1';out.mkdir(exist_ok=False)
    engine=SpatialEngine('multiref-v1');head=RegionalCorrection().to(engine.device);cp=train/'regional/epoch-3.pth'
    assert digest(cp)==run['arms']['regional']['epochs'][2]['checkpoint_sha256'];head.load_state_dict(torch.load(cp,weights_only=True,map_location=engine.device));head.eval().requires_grad_(False)
    report=dict(scope='Post-failure selected-head diagnostic; interventions are not candidates',checkpoint_sha256=digest(cp),training_results_sha256=digest(train/'results.json'),cases=[],finished=False,app_promoted=False,
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['diagnose_regional_correction.py','regional-correction-diagnostic-protocol.md']})
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def read(path,sha=None):
        if sha:assert digest(path)==sha
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    with torch.inference_mode():
        for row in rows:
            a,z=[inputs[row['roles'][r]['feature_index']] for r in ['content','reference']];rgb,ref=[read(ROOT/e['path'],e['parent_sha256']) for e in [a,z]]
            assert digest(cache/a['mask_path'])==a['mask_sha256'];p=np.load(cache/a['mask_path']);prob=torch.from_numpy(p[None,None]).to(engine.device)
            raw=engine.prepare(rgb,ref).float_output(rgb);rawtensor=torch.from_numpy(raw).permute(2,0,1)[None].to(engine.device)
            x=engine.tensor(rgb);c,r=[torch.from_numpy(features[e['index']][None]).to(engine.device) for e in [a,z]]
            outputs={};record=dict(index=row['index'],methods={})
            for method,cc,rr,pp in [('full',c,r,prob),('regions_global',global_control(c),global_control(r),prob),('reference_self',c,c,prob),('probability_half',c,r,torch.full_like(prob,.5))]:
                result=head(x,rawtensor,pp,cc,rr);delta=(result-rawtensor).abs()
                pixels=np.rint(result[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                path=out/f"{row['index']}-{method}.png";Image.fromarray(pixels).save(path);outputs[method]=pixels
                if method=='full':assert np.array_equal(pixels,read(train/'regional/epoch-3'/f"{row['index']}.png"))
                record['methods'][method]=dict(path=path.name,sha256=digest(path),residual_mean_codes=float(delta.mean())*255,residual_max_codes=float(delta.max())*255,saturated_fraction=float((delta>=.099).float().mean()))
            baseline=np.rint(raw.clip(0,1)*255).astype(np.uint8);record['methods']['baseline']=dict(path=f"{row['index']}-baseline.png")
            Image.fromarray(baseline).save(out/record['methods']['baseline']['path']);record['methods']['baseline']['sha256']=digest(out/record['methods']['baseline']['path'])
            rawpath=out/f"{row['index']}-raw.npy";np.save(rawpath,raw,allow_pickle=False);record['raw']=dict(path=rawpath.name,sha256=digest(rawpath));record['probability_index']=a['index']
            report['cases'].append(record)
    manifest={e['original_path']:e for e in json.loads((b/'cohort-multiref-v1/srgb-v1/manifest.json').read_text())['files']}
    for i,record in enumerate(report['cases']):
        entry=manifest[cohort['groups']['validation'][i]['gt']];gt=read(b/'cohort-multiref-v1'/entry['prepared_path'],entry['sha256']).astype(float)
        p=np.load(cache/inputs[record['probability_index']]['mask_path']);raw=np.load(out/record['raw']['path']);basepixels=read(out/record['methods']['baseline']['path']).astype(float)
        full=read(out/record['methods']['full']['path']).astype(float)
        lower=(raw-.1).clip(0,1)*255;upper=(raw+.1).clip(0,1)*255
        record['optimistic_independent_pixel_lower_bound']=float(np.maximum(lower-gt,np.maximum(gt-upper,0)).mean())
        for name,entry in record['methods'].items():
            pixels=read(out/entry['path']).astype(float);error=np.abs(pixels-gt).mean(2)
            entry.update(mae=float(error.mean()),person_weighted_mae=float((error*p).sum()/p.sum()) if p.sum()>0 else None,
                background_weighted_mae=float((error*(1-p)).sum()/(1-p).sum()) if (1-p).sum()>0 else None,
                change_from_full=float(np.abs(pixels-full).mean()),change_from_baseline=float(np.abs(pixels-basepixels).mean()))
        if i<4:
            canvas=Image.new('RGB',(1536,280),'#22252a');d=ImageDraw.Draw(canvas)
            for col,name in enumerate(['baseline','full','regions_global','reference_self','probability_half','gt']):
                im=gt.astype(np.uint8) if name=='gt' else read(out/record['methods'][name]['path']);canvas.paste(Image.fromarray(im),(col*256,24));d.text((col*256+4,4),name,fill='white')
            canvas.save(out/f'review-{i+1}.png')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    evidence={k:v for k,v in report.items() if k!='cases'};evidence['results_sha256']=digest(out/'results.json');evidence['cases']=report['cases']
    (ROOT/'docs/research/2026-11-17-regional-correction-diagnostic-evidence.json').write_text(json.dumps(evidence,indent=2))
    for name in ['baseline','full','regions_global','reference_self','probability_half']:
        print(name,{k:float(np.mean([r['methods'][name][k] for r in report['cases']])) for k in ['mae','change_from_full','change_from_baseline']})
    print('optimistic_bound',np.mean([r['optimistic_independent_pixel_lower_bound'] for r in report['cases']]))


if __name__=='__main__':main()
