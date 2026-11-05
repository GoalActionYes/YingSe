"""External first-pass countertests. No tuning; no mean-target quality score."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import cv2
from PIL import Image,ImageDraw
from check_skin_reference import load,mask,chroma,Y
from candidate_pipeline import baseline
from reference_color_app.lut import apply_mapping
from skin_reference import match_skin

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/content-balance-20261105/skin-external-audit'
def grid(path,panels):
 sheet=Image.new('RGB',(1280,290*max(1,(len(panels)+3)//4)),'#202428');d=ImageDraw.Draw(sheet)
 if not panels:d.text((5,5),'No successful render; inspect errors in results.json',fill='white')
 for i,(label,a) in enumerate(panels):
  im=Image.fromarray(a);im.thumbnail((310,250));x=i%4*320;y=i//4*290
  d.text((x+5,y+5),label,fill='white');sheet.paste(im,(x+(320-im.width)//2,y+30))
 sheet.save(path)
def reference(a,style):
 f=a.astype(float)
 if style=='identity':out=f
 elif style=='warm':out=f*[1.04,.98,.92]+[3,1,0]
 elif style=='cool':out=f*[.92,.98,1.04]+[0,1,3]
 elif style=='desaturate':out=(f@Y)[...,None]+.6*chroma(f)
 elif style=='contrast':out=(f-127.5)*1.12+127.5
 return np.rint(out.clip(0,255)).astype(np.uint8)
def metrics(a,ref,m):
 diff=np.abs(a.astype(float)-ref);s=m>.8
 return dict(skin_rgb_mae=float(diff[s].mean()) if s.any() else None,
   skin_rgb_p95=float(np.quantile(diff[s],.95)) if s.any() else None,
   skin_chroma_mae=float(np.abs(chroma(a)-chroma(ref))[s].mean()) if s.any() else None,
   whole_rgb_mae=float(diff.mean()))
def main():
 started=time.monotonic();data={};records=[];errors=[]
 sources=json.loads((OUT/'sources.json').read_text('utf8'))
 for item in sources:
  if 'error' in item:continue
  name=item['file'];a=load(OUT/'assets'/name);m=mask(a);data[name]=(a,m);panels=[]
  ap=OUT/(Path(name).stem+'-target.png');Image.fromarray(a).save(ap)
  for style in ('identity','warm','cool','desaturate','contrast'):
   ref=reference(a,style);rp=OUT/(Path(name).stem+'-'+style+'-ref.png');Image.fromarray(ref).save(rp)
   try:
    graded=apply_mapping(a,baseline.analyze(ap,rp),1.)
    out,diag=match_skin(graded,ref,m,mask(ref));before=metrics(graded,ref,m);after=metrics(out,ref,m)
    change=out.astype(float)-graded
    record=dict(file=name,style=style,before=before,after=after,diagnostics=diag,
      rgb_mae_change=after['skin_rgb_mae']-before['skin_rgb_mae'] if before['skin_rgb_mae'] is not None else None,
      background_exact=bool(np.array_equal(out[m==0],graded[m==0])),max_luminance_change=float(np.abs(change@Y).max()))
    records.append(record);Image.fromarray(out).save(OUT/(Path(name).stem+'-'+style+'-out.png'))
    _,labels,stats,_=cv2.connectedComponentsWithStats((m>.8).astype(np.uint8),8)
    components=[]
    for component in range(1,len(stats)):
     if stats[component,cv2.CC_STAT_AREA]<32:continue
     selected=labels==component
     components.append(dict(pixels=int(selected.sum()),
      before_rgb_mae=float(np.abs(graded.astype(float)-ref)[selected].mean()),
      after_rgb_mae=float(np.abs(out.astype(float)-ref)[selected].mean())))
    record['skin_components_not_identities']=components
    panels.extend([(name+' target',a),(style+' reference',ref),('actual baseline',graded),('skin residual',out)])
   except (ArithmeticError,ValueError) as exc:errors.append(dict(file=name,style=style,error=str(exc)))
  grid(OUT/(Path(name).stem+'-all-controls.png'),panels)
  print(name+' complete',flush=True)
 pairs=[('obama.jpg','biden.jpg'),('biden.jpg','obama.jpg'),('alex-lacamoire.png','lin-manuel-miranda.png'),
        ('lin-manuel-miranda.png','alex-lacamoire.png'),('two_people.jpg','obama2.jpg'),('obama2.jpg','two_people.jpg')]
 pairs += [('2007_007763.jpg','2008_001322.jpg'),('2008_001322.jpg','2007_007763.jpg'),
           ('2008_002470.jpg','2008_007676.jpg'),('2008_007676.jpg','2008_002470.jpg')]
 natural=[];panels=[]
 for i,(an,bn) in enumerate(pairs):
  if an not in data or bn not in data:continue
  a,m=data[an];b,rm=data[bn]
  try:
   graded=apply_mapping(a,baseline.analyze(OUT/(Path(an).stem+'-target.png'),OUT/(Path(bn).stem+'-target.png')),.8)
   out,diag=match_skin(graded,b,m,rm)
   for label,rgb in (('baseline',graded),('out',out)):Image.fromarray(rgb).save(OUT/f'natural-{i}-{label}.png')
   panels.extend([(f'{i} target {an}',a),('reference '+bn,b),('actual baseline',graded),('skin residual',out)])
   natural.append(dict(case=i,target=an,reference=bn,diagnostics=diag,
    change_mae=float(np.abs(out.astype(float)-graded).mean()),
    note='No objective quality score for unmatched people; author visual review required'))
  except (ArithmeticError,ValueError) as exc:errors.append(dict(target=an,reference=bn,error=str(exc)))
 grid(OUT/'natural-all.png',panels)
 valid=[r for r in records if r['rgb_mae_change'] is not None]
 summary=dict(control_count=len(records),valid_skin_cases=len(valid),improved=sum(r['rgb_mae_change']<-.1 for r in valid),
  worsened=sum(r['rgb_mae_change']>.1 for r in valid),small_change=sum(abs(r['rgb_mae_change'])<=.1 for r in valid),
  max_luminance_change=max((r['max_luminance_change'] for r in records),default=None),
  background_exact=all(r['background_exact'] for r in records),errors=errors)
 report=dict(protocol='skin_external_protocol.md',summary=summary,controlled=records,natural=natural,
  settings=dict(pull=.5,max_rgb_delta=8),product_changed=False,elapsed_seconds=time.monotonic()-started)
 (OUT/'results.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
