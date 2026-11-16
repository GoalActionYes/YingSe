"""Recompute saved PNG metrics without importing the intervention/statistic helper."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from independent_coco import ROOT, digest


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v1'
    base=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    report=json.loads((out/'results.json').read_text())
    assert report['finished'] and len(report['cases'])==6
    assert digest(base/'development/jobs.json')==report['jobs_sha256']
    assert digest(base/'manifest.json')==report['manifest_sha256']
    for n,h in report['source_sha256'].items():
        assert digest(out/'frozen-source'/n)==h
        assert digest(Path(__file__).with_name(n))==h
    checked=0;metrics=0;maximum_error=0.;summaries=[]

    def read(entry):
        nonlocal checked
        p=out/entry['path'];assert digest(p)==entry['sha256']
        with Image.open(p) as im: a=np.array(im.convert('RGB'))
        assert hashlib.sha256(a.tobytes()).hexdigest()==entry['pixel_sha256']
        checked+=1;return a

    for c in report['cases']:
        folder=out/c['pair']
        for p,h in c['parents'].items():assert digest(base/p)==h
        target=read(c['input_file'])
        with Image.open(base/f"{c['pair'].split('-')[0]}-human-mask.png") as im:
            tm=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
        with Image.open(base/f"{c['pair'].split('-')[1]}-human-mask.png") as im:
            rm=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
        assert tm.mean()==c['target_person_fraction'] and rm.mean()==c['reference_person_fraction']
        x=target.astype(float);y=x[...,0]*.299+x[...,1]*.587+x[...,2]*.114
        regions=dict(person=tm,background=~tm,neutral_proxy=(np.ptp(x,axis=2)<=20)&(y>=40)&(y<=220),whole=np.ones(tm.shape,bool))
        assert len(c['variants'])==12
        baseline={};reference=None
        images={}
        for v in c['variants']:
            ref=read(v['reference']);images[('reference',v['name'])]=ref
            if v['name']=='identity':reference=ref.copy()
            if v['name'].startswith('background'):
                assert np.array_equal(ref[rm],reference[rm])
                assert v['intervention']['person_max_change']==0
            summary=dict(pair=c['pair'],group=c['group'],variant=v['name'],aligned=v['certified_aligned'],
                         reference_intervention=v['intervention'],methods={})
            shift=ref.astype(float)-reference.astype(float)
            summary['model_y_mean_reference_delta']=float((shift@np.array([.2126,.7152,.0722])).mean())
            for method,entry in v['methods'].items():
                actual=read(entry['file']);images[(method,v['name'])]=actual
                if v['name']=='identity':baseline[method]=actual.copy()
                if v['name']=='identity_repeat':assert np.array_equal(actual,baseline[method])
                for name,mask in regions.items():
                    r=entry['regions'][name]
                    if not mask.any():assert r is None;continue
                    assert r['pixels']==int(mask.sum())
                    a=actual[mask].astype(float);b=baseline[method][mask].astype(float);d=a-b
                    expected=dict(rgb_mae=float(np.abs(d).mean()),rgb_mean_delta=d.mean(0).tolist(),
                        luminance_mean_delta=float((d[:,0]*.299+d[:,1]*.587+d[:,2]*.114).mean()),
                        saturation_range_mean_delta=float((np.ptp(a,axis=1)-np.ptp(b,axis=1)).mean()))
                    for k,value in expected.items():
                        error=float(np.max(np.abs(np.asarray(value)-r[k])))
                        maximum_error=max(maximum_error,error);assert error<1e-10
                    assert r['sensitive_over_one_code']==(expected['rgb_mae']>1)
                    metrics+=1
                summary['methods'][method]=entry['regions']
            summaries.append(summary)
        # All six pairs, full-coverage +/- interventions, no selection by score.
        canvas=Image.new('RGB',(1024,864),'#22252a');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        for row,method in enumerate(['reference','global','adaptive']):
            for col,name in enumerate(['target','identity','background_green_100','background_magenta_100']):
                rgb=target if col==0 else images[(method,name)]
                canvas.paste(Image.fromarray(rgb),(col*256,row*288+30))
                draw.text((col*256+3,row*288+6),('Target' if col==0 else method+' '+['','identity','green','magenta'][col]),font=font,fill='white')
        canvas.save(folder/'review.png')
    evidence=dict(scope=report['scope'],results_sha256=digest(out/'results.json'),source_sha256=report['source_sha256'],
        audit_script_sha256=digest(Path(__file__)),model_sha256=report['model_sha256'],
        global_fingerprint=report['global_fingerprint'],pairs=6,variant_rows=len(summaries),outputs=144,
        checked_png_records=checked,region_records_independently_recomputed=metrics,maximum_metric_error=maximum_error,
        person_preservation_passed=True,independent_repeat_passed=True,rows=summaries,
        human_evaluations=0,app_promoted=False,quality_gate_passed=False)
    public=ROOT/'docs/research/2026-11-16-background-intervention-evidence.json'
    public.write_text(json.dumps(evidence,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
