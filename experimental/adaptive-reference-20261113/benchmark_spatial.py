"""CPU-only application microbenchmark, not decoder/player/export fps."""
import json
import time
from pathlib import Path
import hashlib
import numpy as np
from spatial import fit,apply,_basis,_design

ROOT=Path(__file__).resolve().parents[2]
a=np.random.default_rng(1).integers(40,210,(96,128,3),dtype=np.uint8)
b=np.rint(a.astype(float)+np.linspace(-20,20,128)[None,:,None]).astype(np.uint8)
model=fit(a,b)
assert model['accepted']
slow=np.rint(np.clip((_design(a,_basis(96,128))@np.asarray(model['coefficients'])).reshape(a.shape)*255,0,255)).astype(np.uint8)
fast=apply(a,model)
max_error=int(np.abs(slow.astype(int)-fast).max())
assert max_error<=1
frame=np.tile(a,(12,16,1))[:1080,:1920]
times=[]
for _ in range(5):
    start=time.perf_counter(); apply(frame,model); times.append(time.perf_counter()-start)
report=dict(shape=list(frame.shape),apply_seconds=times,
            optimized_vs_float64_feature_path_max_code_difference=max_error,
            spatial_source_sha256=hashlib.sha256((Path(__file__).parent/'spatial.py').read_bytes()).hexdigest(),
            scope='CPU application only, coefficients already fitted; no decode, analysis, display or encoding')
out=ROOT/'outputs/adaptive-reference-20261113/spatial-development/benchmark.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
