"""Controlled pixel-pool occupancy sensitivity diagnostic; no product changes."""
from pathlib import Path
import sys,json,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experimental/skin-protection-20261019'))
from reference_color_app import skin,features
from reference_color_app.lut import build_mapping,apply_mapping
OUT=ROOT/'outputs/content-balance-20261105'
target=np.array(Image.open(ROOT/'outputs/region-match-probe/target.png').convert('RGB'))
ref=np.array(Image.open(ROOT/'outputs/region-match-probe/reference-controlled.png').convert('RGB'))
mask=np.array(Image.open(ROOT/'outputs/region-match-probe/person-mask.png'))[...,0]/255
fg=ref[mask>.9];bg=ref[mask<.1]
rng=np.random.default_rng(1105);n=320*320
fpool=fg[rng.integers(0,len(fg),n)];bpool=bg[rng.integers(0,len(bg),n)]
perm=rng.permutation(n);records=[];outputs=[]
start=time.monotonic();source=features.extract_features(target,0)
for share in (.1,.3,.5,.7,.9):
 count=round(n*share);pixels=np.concatenate([fpool[:count],bpool[:n-count]])[perm].reshape(320,320,3)
 mapping=build_mapping(source,features.extract_features(pixels,1));out=apply_mapping(target,mapping,.8)
 Image.fromarray(pixels).save(OUT/f'reference-pool-{int(share*100)}.png')
 Image.fromarray(out).save(OUT/f'output-{int(share*100)}.png');outputs.append(out)
 records.append(dict(person_pool_share=share,output_mean_rgb=out.mean(axis=(0,1)).tolist()))
anchor=outputs[2].astype(float)
for row,out in zip(records,outputs):row['mae_vs_50_percent']=float(np.abs(out.astype(float)-anchor).mean())
report=dict(date='2026-11-05',status='baseline only; no improved method implemented',
 question='How much does a fixed target output change when reference foreground/background pixel occupancy changes?',
 setup='Same two pixel pools; deterministic sampling; 320x320 shuffled atlases; fixed target and strength 0.8',
 caveats=['Pixel atlases are not natural photos','Finite sampling varies across ratios','Does not establish photographic failure rate or aesthetic quality',
 'Original person mask is a model prediction, not human ground truth','No comparison to Jianying or other apps'],
 records=records,elapsed_seconds=time.monotonic()-start)
(OUT/'baseline-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
print(json.dumps(report,indent=2))
