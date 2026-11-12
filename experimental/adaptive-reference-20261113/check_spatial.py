"""Frozen development check for same-layout branch, including explicit rejection controls.

Uses historical photos, not unseen holdout. Reference is also known desired
output only because these tests deliberately use the same scene and layout.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping
from spatial import fit,apply


def main():
    assets=ROOT/'outputs/adaptive-reference-20261113/historical-development'
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-development'
    out.mkdir(parents=True,exist_ok=True)
    photos=[np.asarray(Image.open(assets/f'target-{i}.png').convert('RGB')) for i in range(10)]
    rows=[]
    for i,source in enumerate(photos):
        x=source.astype(float); h,w=x.shape[:2]
        weight=np.linspace(0,1,w)[None,:,None]
        gray=(x @ [.299,.587,.114])[...,None]
        cases=dict(identity=x,uniform=x+12,
                   brightness=x+24*(2*weight-1),
                   contrast=(x-128)*(.8+.4*weight)+128,
                   saturation=gray+(x-gray)*(.5+.6*weight),
                   color=x+(2*weight-1)*np.array([16,0,-16]))
        sourcepath=assets/f'target-{i}.png'
        for name,desired in cases.items():
            desired=np.rint(np.clip(desired,0,255)).astype(np.uint8)
            refpath=out/f'{i}-{name}-reference.png'
            Image.fromarray(desired).save(refpath)
            baseline=apply_mapping(source,analyze(sourcepath,refpath),1.)
            start=time.perf_counter(); model=fit(source,desired)
            fit_seconds=time.perf_counter()-start
            start=time.perf_counter()
            result=apply(source,model) if model['accepted'] else baseline
            apply_seconds=time.perf_counter()-start
            Image.fromarray(result).save(out/f'{i}-{name}-result.png')
            rows.append(dict(photo=i,case=name,accepted=model['accepted'],diagnostics=model,
                             baseline_mae=float(np.abs(baseline.astype(float)-desired).mean()),
                             candidate_mae=float(np.abs(result.astype(float)-desired).mean()),
                             fit_seconds=fit_seconds,apply_seconds=apply_seconds))
        # Resize different content to exact dimensions so size rejection is not enough.
        other=np.asarray(Image.fromarray(photos[(i+1)%10]).resize((w,h),Image.Resampling.BILINEAR))
        rows.append(dict(photo=i,case='different_content',diagnostics=fit(source,other)))
        altered=source.copy()
        altered[h//4:h//2,w//4:w//2]=other[h//4:h//2,w//4:w//2]
        rows.append(dict(photo=i,case='local_replacement',diagnostics=fit(source,altered)))
    results=[r for r in rows if 'candidate_mae' in r]
    negative=[r for r in rows if 'candidate_mae' not in r]
    summary=dict(scope='Ten historical development photos; same-layout branch only, not general reference transfer',
                 positive_cases=len(results),accepted=sum(r['accepted'] for r in results),
                 improved=sum(r['candidate_mae']<r['baseline_mae']-.1 for r in results),
                 worsened=sum(r['candidate_mae']>r['baseline_mae']+.1 for r in results),
                 negative_cases=len(negative),negative_accepted=sum(r['diagnostics']['accepted'] for r in negative),
                 by_case={name:dict(accepted=sum(r['accepted'] for r in results if r['case']==name),
                                   baseline_median=float(np.median([r['baseline_mae'] for r in results if r['case']==name])),
                                   candidate_median=float(np.median([r['candidate_mae'] for r in results if r['case']==name]))) for name in cases})
    (out/'results.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
