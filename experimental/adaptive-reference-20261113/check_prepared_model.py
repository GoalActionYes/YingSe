"""Identity drift and fixed-map reuse checks. Not a motion-video test."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from canonical_model import CanonicalModel
from gpu_mapping import PreparedMapping

ROOT=Path(__file__).resolve().parents[2]
data=ROOT/'outputs/adaptive-reference-20261113/cross-scene'
out=ROOT/'outputs/adaptive-reference-20261113/canonical-model'
model=CanonicalModel()
rows=[]
for i in range(10):
    x=np.asarray(Image.open(data/f'target-{i}.png').convert('RGB'))
    predicted,timing=model.predict(x,x)
    prepared=PreparedMapping(*model.last_luts,device=model.device)
    reused=prepared.apply(x)
    rows.append(dict(photo=i,raw_identity_mae=float(np.abs(predicted.astype(float)-x).mean()),
                     reuse_max_code_difference=int(np.abs(reused.astype(int)-predicted).max()),**timing))
    assert np.abs(reused.astype(int)-predicted).max()<=1
    assert np.array_equal(prepared.apply(x,0),x)
x=np.asarray(Image.open(data/'target-0.png').convert('RGB'))
r=np.asarray(Image.open(data/'reference-1-warm.png').convert('RGB'))
model.predict(x,r)
prepared=PreparedMapping(*model.last_luts,device=model.device)
frame=np.array(Image.fromarray(x).resize((1920,1080)))
times=[]
for _ in range(10):
    start=time.perf_counter();prepared.apply(frame);times.append(time.perf_counter()-start)
report=dict(scope='Raw model identity drift and fixed LUT CPU->GPU->CPU application, not decode/display/video temporal consistency',
            provenance=model.provenance,identity=rows,fixed_1080p_seconds=times,
            fixed_1080p_p50_p95=np.percentile(times,[50,95]).tolist(),
            identity_mae_median=float(np.median([row['raw_identity_mae'] for row in rows])))
(out/'prepared-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('identity','provenance')},indent=2))
