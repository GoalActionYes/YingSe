"""Controlled content proportions and one fixed analysis reweighting experiment."""
import json
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping
from reference_color_app.lut import build_mapping
from reference_color_app.features import extract_features
from reference_color_app.calibration import calibrate


def balanced(rgb,power=0.):
    pixels=rgb.reshape(-1,3);keys=(pixels.astype(np.int32)//16)@np.array([256,16,1])
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    sums=np.stack([np.bincount(inverse,weights=pixels[:,i],minlength=len(unique)) for i in range(3)],1)
    centers=np.rint(sums/counts[:,None]).astype(np.uint8)
    if len(centers)>1024:
        selected=np.linspace(0,len(centers)-1,1024).astype(int);centers=centers[selected];counts=counts[selected]
    weights=counts.astype(float)**power;cdf=np.cumsum(weights)/weights.sum()
    indices=np.searchsorted(cdf,(np.arange(512*512)+.5)/(512*512));np.random.default_rng(20261113).shuffle(indices)
    return centers[indices].reshape(512,512,3),len(unique)


def load(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def small(rgb):
    im=Image.fromarray(rgb);im.thumbnail((640,640),Image.Resampling.BILINEAR);return np.array(im)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--power',type=float,choices=[0.,.5,1.],default=.5);args=parser.parse_args()
    out=ROOT/'outputs/adaptive-reference-20261113/composition'/f'power-{args.power}-v3';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_composition.py','composition-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def predict(target,reference,prefix):
        a,b=out/(prefix+'-analysis-target.png'),out/(prefix+'-analysis-reference.png')
        aa,na=balanced(target,args.power);bb,nb=balanced(reference,args.power);Image.fromarray(aa).save(a);Image.fromarray(bb).save(b)
        start=time.perf_counter();sf,rf=extract_features(aa,0),extract_features(bb,1)
        np.savez_compressed(out/(prefix+'-features.npz'),source=sf.values,reference=rf.values)
        mapping=build_mapping(sf,rf)
        if np.array_equal(aa,bb):mapping=calibrate(mapping,mapping)
        rgb=apply_mapping(target,mapping,1)
        return rgb,dict(nonempty_bins=[na,nb],seconds=time.perf_counter()-start,aligned_fit_disabled='Synthetic analysis proxy cannot certify spatial correspondence')
    colors=np.array([[180,55,50],[60,150,75],[65,105,190],[205,170,65],[145,80,165],[55,160,170],[190,145,110],[110,110,110]],np.uint8)
    def scene(weights):return np.repeat(np.concatenate([np.repeat(c[None,:],int(w)*32,axis=0) for c,w in zip(colors,weights)],axis=0)[None],256,axis=0)
    target=scene([2]*8);controls=[]
    for look in ['identity','warm']:
        def grade(x):return x.copy() if look=='identity' else np.rint(np.clip(x.astype(float)*[1.05,.98,.9]+[2.55,1.275,5.1],0,255)).astype(np.uint8)
        truth=grade(target)
        for n in range(4):
            reference=grade(scene(np.roll([9,1,1,1,1,1,1,1],n*2)))
            prefix=f'control-{look}-{n}';a,b=out/(prefix+'-target.png'),out/(prefix+'-reference.png')
            Image.fromarray(target).save(a);Image.fromarray(reference).save(b)
            raw=apply_mapping(target,analyze(a,b),1);candidate,diag=predict(target,reference,prefix)
            Image.fromarray(raw).save(out/(prefix+'-current.png'));Image.fromarray(candidate).save(out/(prefix+'-balanced.png'))
            controls.append(dict(look=look,variant=n,current_mae=float(np.abs(raw.astype(float)-truth).mean()),balanced_mae=float(np.abs(candidate.astype(float)-truth).mean()),diagnostics=diag))
    data=ROOT/'outputs/adaptive-reference-20261113/pst50';prior=data/'development-canonical-final-v1';rows=[]
    report=dict(scope='Synthetic diagnostic plus consumed development 1-15, not semantic acceptance',balance_power=args.power,controls=controls,cases=rows,
        source_sha256={p.name:digest(p) for p in frozen.iterdir()})
    for i in range(1,16):
        source=data if i<=5 else data/'development-dense-round2-v1/inputs'
        a,b=source/f'content_709/in{i}.png',source/f'paired_style/tar{i}.png'
        row=dict(index=i,source_sha256=dict(target=digest(a),reference=digest(b)));rows.append(row)
        try:
            rgb,diag=predict(load(a),load(b),str(i));path=out/f'{i}-balanced.png';Image.fromarray(small(rgb)).save(path)
            row.update(sha256=digest(path),diagnostics=diag)
        except Exception as exc:row['failure']=dict(type=type(exc).__name__,message=str(exc))
        (out/'prediction-progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(index=i,failure=row.get('failure'),seconds=row.get('diagnostics',{}).get('seconds'))),flush=True)
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        if 'failure' in row:continue
        i=row['index'];source=data if i<=5 else data/'development-dense-round2-v1/inputs'
        if digest(out/f'{i}-balanced.png')!=row['sha256']:raise ValueError('Prediction modified before scoring')
        gt=small(load(source/f'paired_gt/gt{i}.png'));candidate=load(out/f'{i}-balanced.png');current=load(prior/f'{i}-current.png')
        row['scores']={k:float(np.abs(v.astype(np.float32)-gt).mean()) for k,v in [('current',current),('balanced',candidate)]}
        panel=Image.new('RGB',(1080,520),'#222222');draw=ImageDraw.Draw(panel)
        for n,(label,rgb) in enumerate([('Input',small(load(source/f'content_709/in{i}.png'))),('Reference',small(load(source/f'paired_style/tar{i}.png'))),('Expert',gt),('Current',current),('Balanced analysis',candidate)]):
            im=Image.fromarray(rgb);im.thumbnail((356,230));x,y=n%3*360,n//3*260;panel.paste(im,(x,y+25));draw.text((x+8,y+5),label,fill='white')
        panel.save(out/f'{i}-comparison.jpg',quality=94)
    valid=[r for r in rows if 'scores' in r]
    base=np.array([r['scores']['current'] for r in valid]);actual=np.array([r['scores']['balanced'] for r in valid])
    report.update(summary=dict(total_cases=len(rows),scored_cases=len(valid),failed_cases=len(rows)-len(valid),partial_means_if_failed=True,
        current_mean=float(base.mean()),balanced_mean=float(actual.mean()),improved=int((actual<base-.1).sum()),regressed=int((actual>base+.1).sum()),severe=int(((actual>base+5)&(actual>base*1.25)).sum())),
        predictions_sha256=digest(out/'predictions.json'),all_predictions_before_expert_read=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(controls=controls,summary=report['summary'])),flush=True)


if __name__=='__main__':main()
