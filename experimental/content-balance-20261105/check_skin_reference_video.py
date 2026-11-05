"""Short fixed-chroma / moving-mask probe, experimental 30fps with audio."""
from pathlib import Path
import sys,json,subprocess,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from check_skin_reference import ROOT,mask,load,Y
from candidate_pipeline import baseline
from reference_color_app import media,skin
from reference_color_app.lut import apply_mapping
from skin_reference import match_skin,apply_frame
from person_gate import segment

OUT=ROOT/'outputs/content-balance-20261105/skin-external-audit/continuous-video'
OUT.mkdir(parents=True,exist_ok=True)

def main():
 video=ROOT/'outputs/content-balance-20261105/baseline-adapter/with-audio.mp4'
 frames=OUT/'video-frames';frames.mkdir(exist_ok=True);ff=str(media.ffmpeg_path())
 def run(args):subprocess.run([ff,'-hide_banner','-nostdin','-y',*args],check=True,capture_output=True,timeout=180)
 run(['-i',str(video),'-vf','fps=30',str(frames/'input-%04d.png')])
 reference=ROOT/'outputs/content-balance-20261105/holdout-v3/assets/astronaut.png'
 ref=load(reference);rm=mask(ref);mp=baseline.analyze(video,reference)
 first=media.decode_image(frames/'input-0001.png');fm=mask(first)
 _,fit=match_skin(apply_mapping(first,mp,.8),ref,fm,rm);delta=np.array(fit.get('delta_rgb',[0,0,0]))
 session=skin.SkinSession();records=[]
 for index,path in enumerate(sorted(frames.glob('input-*.png')),1):
  a=media.decode_image(path);graded=apply_mapping(a,mp,.8)
  m,gray,diag=skin._predict(a);_,_,face=segment(a)
  if diag['skin_detected'] and face['person_detected']:
   m,state=session._temporal(m,gray);session._previous_gray=gray.copy();session._previous_mask=m.copy()
   full=np.clip(skin._cv.resize(m,(a.shape[1],a.shape[0])),0,1)
  else:session.reset();full=np.zeros(a.shape[:2]);state='no_skin'
  corrected=apply_frame(graded,full,delta,face_eligible=face['person_detected']);change=corrected.astype(float)-graded
  Image.fromarray(corrected).save(frames/f'output-{index:04d}.png')
  assert np.array_equal(corrected[full==0],graded[full==0])
  records.append(dict(frame=index,max_luminance_delta=float(np.abs(change@Y).max()),max_rgb_delta=float(np.abs(change).max()),state=state))
 assert len(records)==90 and max(r['max_luminance_delta'] for r in records)<=.501
 output=OUT/'reference-skin-video.mp4'
 run(['-framerate','30','-i',str(frames/'output-%04d.png'),'-i',str(video),'-map','0:v:0','-map','1:a:0',
      '-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-c:a','aac','-af','apad',
      '-t',format(len(records)/30,'.9f'),str(output)])
 progress=OUT/'video-decode.txt';run(['-xerror','-i',str(output),'-progress',str(progress),'-f','null','-'])
 decoded=int(re.findall(r'^frame=(\d+)\s*$',progress.read_text(),re.M)[-1]);meta=media.probe_media(output)
 assert decoded==90 and meta['has_audio']
 report=dict(frames=90,duration=meta['duration_seconds'],fps=30,audio='Synthetic tone, complete decode; no listening test',
  fitted_delta=fit,records=records,background_exact_before_encode=True,
  limits=['3 seconds, one scene only','No flicker or cut robustness claim','First-frame fit; existing optical-flow mask smoothing reused','Private temporal helper used by probe, not released integration','Lossy video may change background pixels after encode'])
 (OUT/'video-results.json').write_text(json.dumps(report,indent=2),'utf8')
 print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if __name__=='__main__':main()
