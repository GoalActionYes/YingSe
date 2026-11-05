"""Sampling-seed repeatability controls; seeds are NOT independent photographs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as e
import numpy as np,json
target=e.media.decode_image(e.ROOT/'outputs/region-match-probe/target.png')
ref=e.media.decode_image(e.ROOT/'outputs/region-match-probe/reference-controlled.png')
m,_,_=e.segment(target);sm=np.clip(e.fullmask(m,target),0,1)
fg=ref[sm>.9];bg=ref[sm<.1];n=320*320;records=[]
for seed in (1106,1107,1108):
 rng=np.random.default_rng(seed);f=fg[rng.integers(0,len(fg),n)];b=bg[rng.integers(0,len(bg),n)];order=rng.permutation(n)
 scores={}
 for amount in (0,.25,.5):
  outputs=[]
  for p in (.2,.5,.8):
   k=round(n*p);rgb=np.concatenate([f[:k],b[:n-k]])[order].reshape(320,320,3)
   labels=np.concatenate([np.ones(k),np.zeros(n-k)])[order].reshape(320,320)
   mp,_=e.mapping(target,rgb,sm,labels,amount);outputs.append(e.apply_mapping(target,mp,.8))
  anchor=outputs[1].astype(float)
  scores[str(amount)]=float(np.median([np.abs(outputs[i].astype(float)-anchor).mean() for i in (0,2)]))
 records.append(dict(seed=seed,median_drift=scores,reduction={a:1-v/scores['0'] for a,v in scores.items() if a!='0'}))
(e.OUT/'repeat-controls.json').write_text(json.dumps(dict(records=records,warning='Repeated pixel-pool sampling only; no independent natural holdout'),indent=2),'utf8')
print(json.dumps(records,indent=2))
