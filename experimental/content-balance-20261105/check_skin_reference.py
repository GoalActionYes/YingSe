"""Frozen first trial: .5 pull, max8 RGB, 6 controlled + 3 natural pairs."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image,ImageDraw
from candidate_pipeline import baseline
from person_gate import segment,fullmask
from reference_color_app import skin
from reference_color_app.lut import apply_mapping
from skin_reference import match_skin

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/content-balance-20261105/reference-skin-residual';OUT.mkdir(exist_ok=True)
Y=np.array([.299,.587,.114])
def chroma(a):
 a=a.astype(float);return a-(a@Y)[...,None]
def load(path):
 im=Image.open(path).convert('RGB');im.thumbnail((640,640));return np.asarray(im)
def mask(a):
 _,_,diag=segment(a);m,_,_=skin._predict(a)
 return np.clip(fullmask(m,a),0,1) if diag['person_detected'] else np.zeros(a.shape[:2])
def main():
 old=ROOT/'outputs/content-balance-20261105/holdout-v3/assets';new=ROOT/'outputs/content-balance-20261105/person-gate-v4/assets'
 paths=[old/'astronaut.png',old/'grace_hopper.jpg',old/'male_full_height_hands.jpg',new/'lena.jpg',new/'messi5.jpg',ROOT/'experimental/skin-protection-20261019/reference_color_app/demo/skin-target.png']
 photos=[];records=[];panels=[]
 for path in paths:
  a=load(path);m=mask(a);photos.append((a,m))
  warm=np.array([8.,-2.,-10.]);warm-=warm@Y
  cool=np.array([-8.,1.,6.]);cool-=cool@Y
  ref=np.rint(np.clip(a+m[...,None]*warm+(1-m[...,None])*cool,0,255)).astype(np.uint8)
  rp=OUT/(path.stem+'-reference.png');ap=OUT/(path.stem+'-target.png');Image.fromarray(ref).save(rp);Image.fromarray(a).save(ap)
  graded=apply_mapping(a,baseline.analyze(ap,rp),1.0);rm=mask(ref)
  corrected,diag=match_skin(graded,ref,m,rm)
  selected=m>.8
  errors=[float(np.abs(chroma(rgb)-chroma(ref))[selected].mean()) if selected.any() else None for rgb in (graded,corrected)]
  change=corrected.astype(float)-graded
  records.append(dict(file=path.name,skin_chroma_mae=errors,reduction=1-errors[1]/errors[0] if errors[0] else None,
   max_luminance_change=float(np.abs(change@Y).max()),background_exact=bool(np.array_equal(corrected[m==0],graded[m==0])),
   max_rgb_change=float(np.abs(change).max()),diagnostics=diag))
  panels.extend([(path.stem+' target',a),('controlled reference',ref),('actual baseline',graded),('skin residual',corrected)])
  Image.fromarray(corrected).save(OUT/(path.stem+'-result.png'))
 natural=[]
 for i,j in ((0,1),(3,4),(4,0)):
  a,m=photos[i];ref,rm=photos[j];ap=OUT/f'natural-{i}-target.png';rp=OUT/f'natural-{i}-reference.png'
  Image.fromarray(a).save(ap);Image.fromarray(ref).save(rp)
  graded=apply_mapping(a,baseline.analyze(ap,rp),.8);corrected,diag=match_skin(graded,ref,m,rm)
  panels.extend([(f'natural {i} target',a),('natural reference',ref),('actual baseline',graded),('skin residual',corrected)])
  natural.append(dict(target=i,reference=j,diagnostics=diag,note='No style ground truth; author visual inspection only'))
 sheet=Image.new('RGB',(1280,300*((len(panels)+3)//4)),'#202428');draw=ImageDraw.Draw(sheet)
 for i,(label,a) in enumerate(panels):
  im=Image.fromarray(a);im.thumbnail((310,260));x=i%4*320;y=i//4*300
  draw.text((x+5,y+4),label,fill='white');sheet.paste(im,(x+(320-im.width)//2,y+30))
 sheet.save(OUT/'comparison.png')
 report=dict(controlled=records,natural=natural,settings=dict(pull=.5,max_rgb_delta=8),
  product_changed=False,limits=['Control styles generated using predicted masks; not independent style truth','No video validation','Not representable by global CUBE','No human blind review'])
 (OUT/'results.json').write_text(json.dumps(report,indent=2),'utf8')
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
