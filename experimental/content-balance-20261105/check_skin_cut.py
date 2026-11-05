"""Person -> cat -> person cut countertest for frame eligibility and reset."""
from pathlib import Path
import sys,json,subprocess,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image,ImageOps
from check_skin_reference import ROOT,Y
from reference_color_app import skin,media,pipeline
from reference_color_app.lut import apply_mapping
from person_gate import segment
from skin_reference import apply_frame

OUT=ROOT/'outputs/content-balance-20261105/skin-external-audit'
def main():
 frames=OUT/'cut-frames';frames.mkdir(exist_ok=True)
 source=ROOT/'outputs/content-balance-20261105/baseline-adapter/with-audio.mp4'
 ref=ROOT/'outputs/content-balance-20261105/holdout-v3/assets/astronaut.png'
 fitted=json.loads((ROOT/'outputs/content-balance-20261105/reference-skin-residual/video-results.json').read_text())['fitted_delta']
 delta=np.array(fitted['delta_rgb']);mapping=pipeline.analyze(source,ref);session=skin.SkinSession();records=[]
 person=ROOT/'outputs/content-balance-20261105/reference-skin-residual/video-frames'
 cat=np.array(ImageOps.pad(Image.open(ROOT/'outputs/content-balance-20261105/holdout-v3/assets/cat.jpg').convert('RGB'),(640,360)))
 for i in range(90):
  is_cat=30<=i<60
  a=cat if is_cat else media.decode_image(person/f'input-{i%30+1:04d}.png')
  graded=apply_mapping(a,mapping,.8);m,gray,diag=skin._predict(a);_,_,face=segment(a)
  if face['person_detected'] and diag['skin_detected']:
   m,state=session._temporal(m,gray);session._previous_gray=gray.copy();session._previous_mask=m.copy()
   full=np.clip(skin._cv.resize(m,(640,360)),0,1)
  else:session.reset();full=np.zeros((360,640));state='no_eligible_face'
  out=apply_frame(graded,full,delta,face_eligible=face['person_detected'])
  if is_cat:assert np.array_equal(out,graded)
  change=out.astype(float)-graded;assert np.abs(change@Y).max()<=.501
  Image.fromarray(out).save(frames/f'output-{i+1:04d}.png')
  records.append(dict(frame=i+1,is_cat=is_cat,state=state,face_eligible=face['person_detected'],
    changed_pixels=int(np.any(out!=graded,axis=2).sum()),max_luminance_delta=float(np.abs(change@Y).max())))
 assert records[60]['state']=='new_history'
 ff=str(media.ffmpeg_path());output=OUT/'person-cat-person.mp4'
 def run(args):subprocess.run([ff,'-hide_banner','-nostdin','-y',*args],check=True,capture_output=True,timeout=150)
 run(['-framerate','30','-i',str(frames/'output-%04d.png'),'-i',str(source),'-map','0:v:0','-map','1:a:0',
  '-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-c:a','aac','-af','apad','-t','3',str(output)])
 progress=OUT/'cut-decode.txt';run(['-xerror','-i',str(output),'-progress',str(progress),'-f','null','-'])
 decoded=int(re.findall(r'^frame=(\d+)\s*$',progress.read_text(),re.M)[-1]);assert decoded==90
 report=dict(frames=90,cat_frames_exact_fallback=True,person_return_resets_history=True,
  has_audio=media.probe_media(output)['has_audio'],records=records,
  limits=['Synthetic cut sequence, not independent camera footage','Pre-encode exact fallback, not lossy pixel equality','No universal flicker claim'])
 (OUT/'cut-results.json').write_text(json.dumps(report,indent=2),'utf8');print('Cut probe passed: 90 frames, 30 cat frames exact fallback, history reset on person return')
if __name__=='__main__':main()
