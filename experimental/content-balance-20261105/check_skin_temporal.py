"""Motion-aligned correction-change diagnostic; not a perceptual flicker score."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,cv2
from check_skin_reference import ROOT
from reference_color_app import media,pipeline,skin
from reference_color_app.lut import apply_mapping
from person_gate import segment,fullmask
from skin_reference import apply_frame

OUT=ROOT/'outputs/content-balance-20261105/skin-external-audit'
def main():
 base=OUT/'continuous-video';report=json.loads((base/'video-results.json').read_text());delta=np.array(report['fitted_delta']['delta_rgb'])
 mapping=pipeline.analyze(ROOT/'outputs/content-balance-20261105/baseline-adapter/with-audio.mp4',ROOT/'outputs/content-balance-20261105/holdout-v3/assets/astronaut.png')
 previous=None;rows=[]
 for i in range(1,91):
  a=media.decode_image(base/'video-frames'/f'input-{i:04d}.png');graded=apply_mapping(a,mapping,.8)
  smooth=media.decode_image(base/'video-frames'/f'output-{i:04d}.png')
  m,_,_=skin._predict(a);_,_,d=segment(a);raw=apply_frame(graded,fullmask(m,a),delta,face_eligible=d['person_detected'])
  gray=cv2.resize(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),(160,90))
  corrections=[cv2.resize(x.astype(np.float32)-graded,(160,90)) for x in (raw,smooth)]
  if previous is not None:
   pg,pc=previous;flow=cv2.calcOpticalFlowFarneback(gray,pg,None,.5,3,15,3,5,1.2,0)
   yy,xx=np.mgrid[:90,:160].astype(np.float32);mx,my=xx+flow[...,0],yy+flow[...,1]
   oldgray=cv2.remap(pg,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
   trust=(np.abs(gray.astype(float)-oldgray)<20)&(mx>=0)&(mx<159)&(my>=0)&(my<89)
   row=dict(frame=i)
   for label,now,old in zip(('raw','smoothed'),corrections,pc):
    warped=cv2.remap(old,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    active=trust&((np.abs(now).max(axis=2)>.5)|(np.abs(warped).max(axis=2)>.5))
    diff=np.abs(now-warped).mean(axis=2)[active]
    row[label]=dict(active_pixels=int(active.sum()),mean_change=float(diff.mean()) if diff.size else None,p95=float(np.quantile(diff,.95)) if diff.size else None)
   rows.append(row)
  previous=(gray,corrections)
 summary={label:dict(median_frame_mean_change=float(np.median([r[label]['mean_change'] for r in rows if r[label]['mean_change'] is not None])),
   max_frame_p95=float(max(r[label]['p95'] for r in rows if r[label]['p95'] is not None))) for label in ('raw','smoothed')}
 result=dict(summary=summary,pairs=len(rows),records=rows,units='8bit RGB correction-code change, downsampled 160x90, active photometrically trusted motion-aligned pixels',
  limits=['One 3-second clip','Flow and segmentation errors included','No human-visible flicker threshold','Not a natural-camera benchmark'])
 (OUT/'temporal-diagnostic.json').write_text(json.dumps(result,indent=2),'utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
