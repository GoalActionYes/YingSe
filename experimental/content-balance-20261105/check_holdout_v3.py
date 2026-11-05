"""Run frozen holdout_v3_protocol.md; no parameter selection from results."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import evaluate as e
from check_natural import load,mask
from composition import extract_pair,reliable_share

OUT=e.ROOT/'outputs/content-balance-20261105/holdout-v3'

ERRORS=[]

def render(a,b,am,bm,amount,context):
 x,y,d=extract_pair(a,b,am,bm,amount)
 try:return e.apply_mapping(a,e.build_mapping(x,y),.8),d
 except ArithmeticError as error:
  # Do not hide unsupported native numeric branches with an invented fallback.
  np.savez(OUT/(context+'-packets.npz'),source=x.values,reference=y.values)
  failure=dict(context=context,error=str(error),diagnostics=d)
  ERRORS.append(failure)
  return None,failure

def main():
 ERRORS.clear();started=time.monotonic();records=[];panels=[];occupancy=[]
 for name in ('astronaut.png','grace_hopper.jpg','male_full_height_hands.jpg','cat.jpg'):
  a=load(OUT/'assets'/name);am,ad=mask(a)
  desired=np.rint(np.clip(a.astype(float)*[1.04,.98,.92]+[3,1,0],0,255)).astype(np.uint8)
  bm,bd=mask(desired);record=dict(file=name,target_mask=ad,reference_mask=bd,controlled=[],occupancy=None)
  panels.extend([(name+' target',a),(name+' reference',desired)])
  for amount in (0,.5):
   out,diag=render(a,desired,am,bm,amount,Path(name).stem+f'-control-{amount}')
   if out is None:
    record['controlled'].append(dict(amount=amount,error=diag));continue
   Image.fromarray(out).save(OUT/(Path(name).stem+f'-amount-{amount}.png'))
   record['controlled'].append(dict(amount=amount,mae=float(np.abs(out.astype(float)-desired).mean()),
     boundary_fraction=float(((out==0)|(out==255)).any(axis=2).mean()),diagnostics=diag))
   panels.append((name+f' amount {amount}',out))
  if name!='cat.jpg' and reliable_share(a,am) is not None:
   # Every quota repeats exactly the SAME 1000 colors in each group.
   rng=np.random.default_rng(1105);fg=desired[am>.8];bg=desired[am<.2]
   f=fg[rng.integers(0,len(fg),1000)];b=bg[rng.integers(0,len(bg),1000)]
   scores={};rows=[]
   for amount in (0,.5):
    outputs=[]
    for n in (2,5,8):
     ref=np.concatenate([np.tile(f,(n,1)),np.tile(b,(10-n,1))]).reshape(100,100,3)
     labels=np.concatenate([np.ones(1000*n),np.zeros(1000*(10-n))]).reshape(100,100)
     out,diag=render(a,ref,am,labels,amount,Path(name).stem+f'-occupancy-{amount}-{n}')
     outputs.append(out)
     if out is not None:rows.append((name+f' a{amount} p{n/10}',out))
    scores[str(amount)]=float(np.median([np.abs(outputs[i].astype(float)-outputs[1].astype(float)).mean() for i in (0,2)])) if all(o is not None for o in outputs) else None
   reduction=1-scores['0.5']/scores['0'] if scores['0'] and scores['0.5'] is not None else None
   record['occupancy']=dict(median_drift=scores,reduction=reduction,exact_within_region_palette=True)
   occupancy.extend(rows)
  records.append(record)
 e.OUT=OUT;e.grid('controlled-holdout.png',panels);e.grid('occupancy-holdout.png',occupancy)
 human=records[:3]
 gate=dict(three_occupancy_groups=all(r['occupancy'] is not None for r in human),
   occupancy_each_at_least_20_percent=all(r['occupancy'] is not None and r['occupancy']['reduction'] is not None and r['occupancy']['reduction']>=.2 for r in human),
   controlled_error_increase_at_most_point2=all(all('mae' in c for c in r['controlled']) and r['controlled'][1]['mae']-r['controlled'][0]['mae']<=.2 for r in human),
   no_unsupported_numeric_branches=not ERRORS,
   aesthetic_blind_review_completed=False)
 report=dict(protocol='holdout_v3_protocol.md',research_only=True,product_changed=False,records=records,gate=gate,errors=ERRORS,
   note='Synthetic compositions derived from three new natural photos; not independent natural reference styles. No tuning on these results.',
   elapsed_seconds=time.monotonic()-started)
 (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
