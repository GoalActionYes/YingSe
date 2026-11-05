"""Actual raw vs self-calibrated map countertests. Imports owning root package."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from reference_color_app.features import extract_features
from reference_color_app.lut import build_mapping,apply_mapping
from reference_color_app.calibration import calibrate

DATA=ROOT/'outputs/content-balance-20261105/skin-external-audit'
OUT=DATA/'neutral-calibration';OUT.mkdir(exist_ok=True)
Y=np.array([.299,.587,.114])
def load(p):
 im=Image.open(p).convert('RGB');im.thumbnail((640,640));return np.asarray(im)
def ref(a,style):
 f=a.astype(float)
 if style=='identity':return a
 if style=='near':return np.minimum(a.astype(int)+1,255).astype(np.uint8)
 if style=='warm':f=f*[1.04,.98,.92]+[3,1,0]
 if style=='cool':f=f*[.92,.98,1.04]+[0,1,3]
 if style=='desaturate':f=(f@Y)[...,None]+.6*(f-(f@Y)[...,None])
 if style=='contrast':f=(f-127.5)*1.12+127.5
 return np.rint(f.clip(0,255)).astype(np.uint8)
def main():
 if len(sys.argv)==1:
  records=[];errors=[]
  for row in json.loads((DATA/'sources.json').read_text()):
   if 'error' in row:continue
   subprocess.run([sys.executable,'-I','-B',__file__,row['file']],check=True)
   part=json.loads((OUT/(Path(row['file']).stem+'-results.json')).read_text())
   records+=part['records'];errors+=part['errors']
  write_report(records,errors,OUT/'results.json');return
 records=[];errors=[]
 for row in json.loads((DATA/'sources.json').read_text()):
  if 'error' in row or row['file']!=sys.argv[1]:continue
  a=load(DATA/'assets'/row['file']);source=extract_features(a,0);self_ref=extract_features(a,1)
  neutral=build_mapping(source,self_ref);panels=[]
  for style in ('identity','near','warm','cool','desaturate','contrast'):
   b=ref(a,style)
   try:
    raw=neutral if style=='identity' else build_mapping(source,extract_features(b,1));fixed=calibrate(raw,neutral)
    before=apply_mapping(a,raw,1);after=apply_mapping(a,fixed,1)
    metrics=dict(file=row['file'],style=style,raw_mae=float(np.abs(before.astype(float)-b).mean()),
      fixed_mae=float(np.abs(after.astype(float)-b).mean()),max_change_from_source=int(np.abs(after.astype(int)-a).max()))
    records.append(metrics);panels.extend([(style+' target',a),('reference',b),('raw reconstruction',before),('calibrated',after)])
   except (ArithmeticError,ValueError) as ex:errors.append(dict(file=row['file'],style=style,error=str(ex)))
  im=Image.new('RGB',(1280,290*6),'#202428');d=ImageDraw.Draw(im)
  for i,(label,rgb) in enumerate(panels):
   thumb=Image.fromarray(rgb);thumb.thumbnail((310,250));x=i%4*320;y=i//4*290
   d.text((x+5,y+5),label,fill='white');im.paste(thumb,(x+(320-thumb.width)//2,y+30))
  im.save(OUT/(Path(row['file']).stem+'-comparison.png'));print(row['file']+' complete',flush=True)
 write_report(records,errors,OUT/(Path(sys.argv[1]).stem+'-results.json'))

def write_report(records,errors,path):
 summary={}
 for style in ('identity','near','warm','cool','desaturate','contrast'):
  group=[r for r in records if r['style']==style]
  summary[style]=dict(cases=len(group),raw_median=float(np.median([r['raw_mae'] for r in group])),fixed_median=float(np.median([r['fixed_mae'] for r in group])),
   worsened=sum(r['fixed_mae']>r['raw_mae']+.1 for r in group),max_identity_error=max(r['max_change_from_source'] for r in group) if style=='identity' else None)
 result=dict(summary=summary,records=records,errors=errors,limits=['Known synthetic whole-image styles, not independent natural style truth','Previously used test images; no untouched holdout claim'])
 path.write_text(json.dumps(result,indent=2),'utf8')
 if path.name=='results.json':print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
