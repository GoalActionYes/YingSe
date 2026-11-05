"""Frozen correspondence trial: 12 photos x8 cases and 132 unequal pairs."""
from pathlib import Path
import sys,json,io
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
from PIL import Image,ImageDraw
from aligned_reference import fit_aligned,make_mapping
from reference_color_app import pipeline,media
from reference_color_app.lut import apply_mapping,build_mapping
from reference_color_app.features import extract_features
from reference_color_app.calibration import calibrate

DATA=ROOT/'outputs/content-balance-20261105/skin-external-audit';OUT=DATA/'aligned-trial';OUT.mkdir(exist_ok=True)
Y=np.array([.299,.587,.114])
def load(p):
 im=Image.open(p).convert('RGB');im.thumbnail((640,640));return np.asarray(im)
def main():
 names=[r['file'] for r in json.loads((DATA/'sources.json').read_text()) if 'error' not in r]
 photos={n:load(DATA/'assets'/n) for n in names};negatives=[]
 for a in names:
  for b in names:
   if a!=b:negatives.append(dict(target=a,reference=b,fit=fit_aligned(photos[a],photos[b])))
 records=[];errors=[]
 for name,a in photos.items():
  ap=OUT/(Path(name).stem+'-source.png');Image.fromarray(a).save(ap);f=a.astype(float);panels=[]
  styles=dict(identity=a,near=np.minimum(a.astype(int)+1,255).astype(np.uint8),
   warm=np.rint((f*[1.04,.98,.92]+[3,1,0]).clip(0,255)).astype(np.uint8),
   cool=np.rint((f*[.92,.98,1.04]+[0,1,3]).clip(0,255)).astype(np.uint8),
   desaturate=np.rint(((f@Y)[...,None]+.6*(f-(f@Y)[...,None])).clip(0,255)).astype(np.uint8),
   contrast=np.rint(((f-127.5)*1.12+127.5).clip(0,255)).astype(np.uint8))
  jpeg=io.BytesIO();Image.fromarray(styles['warm']).save(jpeg,format='JPEG',quality=92)
  styles['jpeg92']=np.asarray(Image.open(io.BytesIO(jpeg.getvalue())).convert('RGB'))
  styles['halfsize']=np.asarray(Image.fromarray(styles['warm']).resize((a.shape[1]//2,a.shape[0]//2),Image.Resampling.BILINEAR))
  for style,ref in styles.items():
   rp=OUT/(Path(name).stem+'-'+style+'-reference.png');Image.fromarray(ref).save(rp)
   fit=fit_aligned(a,ref)
   try:
    raw=build_mapping(extract_features(a,0),extract_features(ref,1))
    baseline=calibrate(raw,raw) if np.array_equal(a,ref) else raw
    before=apply_mapping(a,baseline,1)
    after=apply_mapping(a,make_mapping(fit),1) if fit['accepted'] else before
    truth=styles['warm'] if style=='halfsize' else ref
    record=dict(file=name,style=style,fit=fit,baseline_mae=float(np.abs(before.astype(float)-truth).mean()),
     candidate_mae=float(np.abs(after.astype(float)-truth).mean()),max_rgb_error=int(np.abs(after.astype(int)-truth).max()))
    records.append(record);panels.extend([(style+' target',a),('reference',ref),('current baseline',before),('certified candidate',after)])
   except (ValueError,ArithmeticError) as ex:errors.append(dict(file=name,style=style,error=str(ex)))
  im=Image.new('RGB',(1280,290*8),'#202428');d=ImageDraw.Draw(im)
  for i,(label,rgb) in enumerate(panels):
   thumb=Image.fromarray(rgb);thumb.thumbnail((310,250));x=i%4*320;y=i//4*290
   d.text((x+5,y+5),label,fill='white');im.paste(thumb,(x+(320-thumb.width)//2,y+30))
  im.save(OUT/(Path(name).stem+'-comparison.png'));print(name+' done',flush=True)
 summary={}
 for style in styles:
  rows=[r for r in records if r['style']==style]
  summary[style]=dict(cases=len(rows),accepted=sum(r['fit']['accepted'] for r in rows),baseline_median=float(np.median([r['baseline_mae'] for r in rows])),
   candidate_median=float(np.median([r['candidate_mae'] for r in rows])),regressions=sum(r['candidate_mae']>r['baseline_mae']+.1 for r in rows))
 result=dict(summary=summary,records=records,negative_pairs=negatives,false_accepts=[r for r in negatives if r['fit']['accepted']],errors=errors,
  limits=['Previously used photos; not untouched holdout','Same composition support only','No general different-content improvement claim','No true-positive compression/resize coverage guarantee'])
 (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(json.dumps(dict(summary=summary,false_accepts=len(result['false_accepts']),errors=errors),indent=2))
if __name__=='__main__':main()
