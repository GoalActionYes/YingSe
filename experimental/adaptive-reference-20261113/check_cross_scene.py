"""Frozen cross-scene counter-test; see cross_scene_protocol.md before interpreting."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping
from regional import fit,apply
from segment import Segmenter


def transform(rgb,name):
    x=rgb.astype(float)
    if name=='warm':x=x*[1.04,.98,.92]+[3,1,0]
    elif name=='cool':x=x*[.92,.98,1.04]+[0,1,3]
    elif name=='contrast':x=(x-127.5)*1.12+127.5
    elif name=='desaturate':
        gray=(x @ [.299,.587,.114])[...,None]
        x=gray+(x-gray)*.6
    elif name!='identity':raise ValueError(name)
    return np.rint(np.clip(x,0,255)).astype(np.uint8)


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/cross-scene'
    out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'outputs/adaptive-reference-20261113/historical-development'
    arrays=[]
    for i in range(10):
        image=Image.open(data/f'target-{i}.png').convert('RGB')
        image.thumbnail((480,480),Image.Resampling.BILINEAR)
        arrays.append(np.array(image))
    names=('identity','warm','cool','contrast','desaturate')
    engine=Segmenter()
    rows=[]
    try:
        targets=[engine.predict(x)[0] for x in arrays]
        refs={}
        for i,x in enumerate(arrays):
            Image.fromarray(x).save(out/f'target-{i}.png')
            for name in names:
                ref=transform(x,name)
                refs[i,name]=(ref,engine.predict(ref)[0])
                Image.fromarray(ref).save(out/f'reference-{i}-{name}.png')
        # Append each result immediately; process handle remains authority for
        # completion. This file is progress evidence, not a restart signal.
        with (out/'progress.jsonl').open('w',encoding='utf-8') as stream:
            for i,x in enumerate(arrays):
                for step in (1,3,5):
                    j=(i+step)%10
                    for name in names:
                        start=time.perf_counter()
                        ref,rmask=refs[j,name]
                        baseline=apply_mapping(x,analyze(out/f'target-{i}.png',out/f'reference-{j}-{name}.png'),1.)
                        model=fit(baseline,ref,targets[i],rmask)
                        candidate=apply(baseline,targets[i],model)
                        desired=transform(x,name)
                        eb=np.abs(baseline.astype(float)-desired)
                        ec=np.abs(candidate.astype(float)-desired)
                        skin=targets[i][:,:,2]+targets[i][:,:,3]>.8
                        row=dict(target=i,reference=j,style=name,baseline_mae=float(eb.mean()),
                                 candidate_mae=float(ec.mean()),
                                 skin_baseline_mae=float(eb[skin].mean()) if skin.any() else None,
                                 skin_candidate_mae=float(ec[skin].mean()) if skin.any() else None,
                                 non_skin_baseline_mae=float(eb[~skin].mean()) if (~skin).any() else None,
                                 non_skin_candidate_mae=float(ec[~skin].mean()) if (~skin).any() else None,
                                 candidate_clipped_fraction=float(np.any((candidate==0)|(candidate==255),axis=2).mean()),
                                 active_regions=sum(r['active'] for r in model['regions']),
                                 seconds=time.perf_counter()-start)
                        Image.fromarray(baseline).save(out/f'{i}-{j}-{name}-global.png')
                        Image.fromarray(candidate).save(out/f'{i}-{j}-{name}-candidate.png')
                        rows.append(row)
                        stream.write(json.dumps(row)+'\n');stream.flush()
        def summary(group):
            return dict(count=len(group),improved=sum(r['candidate_mae']<r['baseline_mae']-.1 for r in group),
                        worsened=sum(r['candidate_mae']>r['baseline_mae']+.1 for r in group),
                        median_baseline_mae=float(np.median([r['baseline_mae'] for r in group])),
                        median_candidate_mae=float(np.median([r['candidate_mae'] for r in group])))
        report=dict(scope='Historical development counter-test, synthetic global transform across different photos',
                    source_sha256=hashlib.sha256((Path(__file__).parent/'regional.py').read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256((Path(__file__).parent/'cross_scene_protocol.md').read_bytes()).hexdigest(),
                    summary=summary(rows),by_style={name:summary([r for r in rows if r['style']==name]) for name in names},cases=rows)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    finally:engine.close()


if __name__=='__main__':main()
