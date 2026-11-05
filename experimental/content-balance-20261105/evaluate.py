"""Reproducible controlled comparison + real-photo illustrations, not acceptance."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path[:0]=[str(HERE),str(ROOT/'experimental/skin-protection-20261019'),str(ROOT/'work/region-match-probe')]
import numpy as np
from PIL import Image,ImageDraw
from balance import sample_atlas
from reference_color_app import features,media
from reference_color_app.lut import build_mapping,apply_mapping
from run_probe import segment,fullmask
OUT=ROOT/'outputs/content-balance-20261105/candidate';OUT.mkdir(exist_ok=True)

def mapping(source,ref,sm,rm,amount):
 a,sd=sample_atlas(source,sm,amount);b,rd=sample_atlas(ref,rm,amount)
 return build_mapping(features.extract_features(a,0),features.extract_features(b,1)),dict(source=sd,reference=rd)

def grid(name,panels):
 w,h=400,275;im=Image.new('RGB',(w*3,h*int(np.ceil(len(panels)/3))),'#171e24');d=ImageDraw.Draw(im)
 for i,(label,rgb) in enumerate(panels):
  p=Image.fromarray(rgb);p.thumbnail((w,h-28));x=i%3*w;y=i//3*h;d.text((x+10,y+6),label,fill='white');im.paste(p,(x+(w-p.width)//2,y+28+(h-28-p.height)//2))
 im.save(OUT/name)

def main():
 started=time.monotonic();target=media.decode_image(ROOT/'outputs/region-match-probe/target.png')
 ref=media.decode_image(ROOT/'outputs/region-match-probe/reference-controlled.png');m,_,_=segment(target);sm=np.clip(fullmask(m,target),0,1)
 fg=ref[sm>.9];bg=ref[sm<.1];rng=np.random.default_rng(1105);n=320*320
 f=fg[rng.integers(0,len(fg),n)];b=bg[rng.integers(0,len(bg),n)];order=rng.permutation(n)
 runs={};panels=[];reference_panels=[]
 for amount in (0,.25,.5):
  records=[];outputs=[]
  for p in (.1,.2,.5,.8,.9):
   k=round(n*p);rgb=np.concatenate([f[:k],b[:n-k]])[order].reshape(320,320,3)
   labels=np.concatenate([np.ones(k),np.zeros(n-k)])[order].reshape(320,320)
   mp,diag=mapping(target,rgb,sm,labels,amount);out=apply_mapping(target,mp,.8)
   outputs.append(out);Image.fromarray(out).save(OUT/f'controlled-a{amount}-p{p}.png')
   records.append(dict(person_share=p,diagnostics=diag))
  anchor=outputs[2].astype(float)
  for r,out in zip(records,outputs):r['mae_vs_50_percent']=float(np.abs(out.astype(float)-anchor).mean())
  runs[str(amount)]=records
  for i in (1,2,3):panels.append((f'amount {amount} / person {records[i]["person_share"]}',outputs[i]))
 grid('controlled-comparison.png',panels)
 score={}
 for amount,records in runs.items():score[amount]=float(np.median([r['mae_vs_50_percent'] for r in records if r['person_share']!=.5]))
 reductions={a:1-v/score['0'] for a,v in score.items() if a!='0'}
 # Different-content real reference uses actual model masks; show all amounts without selecting a winner.
 other=media.decode_image(ROOT/'outputs/region-match-probe/different-content-reference.png');om,_,_=segment(other);rm=np.clip(fullmask(om,other),0,1)
 natural=[]
 reference_panels=[('Target',target),('Different-content reference',other),('Shipped global baseline',apply_mapping(target,build_mapping(features.extract_features(target,0),features.extract_features(other,1)),.8))]
 for amount in (0,.25,.5):
  mp,diag=mapping(target,other,sm,rm,amount);out=apply_mapping(target,mp,.8)
  reference_panels.append((f'Same sampler / amount {amount}',out));natural.append(dict(amount=amount,diagnostics=diag))
 grid('real-photo-comparison.png',reference_panels)
 # Same-image reference tests reveal drift; no claimed identity property at nonzero grade strength.
 identity=[]
 for amount in (0,.25,.5):
  mp,_=mapping(target,target,sm,sm,amount);out=apply_mapping(target,mp,.8)
  identity.append(dict(amount=amount,mae=float(np.abs(out.astype(float)-target).mean())))
 report=dict(research_only=True,product_changed=False,controlled_runs=runs,median_drift=score,reduction_vs_same_sampler=reductions,
  real_photo_runs=natural,same_image_reference=identity,elapsed_seconds=time.monotonic()-started,
  limitations=['One pixel-pool pair; oracle labels on synthetic atlas','Two official photographs only, no independent holdout dataset',
  'Masks are model predictions on real photos; no manual ground truth','No new video candidate, blind review or aesthetic quality validation',
  'Reduction relative to own 50 percent output measures invariance, not accurate style reproduction'],
  next_gate='Do not ship until natural-photo coherence and independent holdout tests pass')
 (OUT/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(dict(median_drift=score,reduction=reductions,same_image_reference=identity),indent=2))

if __name__=='__main__':main()
