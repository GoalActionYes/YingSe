"""Additional photos + fixed-map video engineering check, not aesthetic approval."""
from pathlib import Path
import sys,json,subprocess,re,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as e
from dense_features import extract_dense
from reference_color_app.lut import cube_text
import numpy as np
from PIL import Image

OUT=e.ROOT/'outputs/content-balance-20261105/natural-check';OUT.mkdir(exist_ok=True)
def fit(a,b,am,bm,amount):return e.build_mapping(extract_dense(a,0,am,amount),extract_dense(b,1,bm,amount))
def load(p):
 im=Image.open(p).convert('RGB');im.thumbnail((640,640));return np.asarray(im)
def mask(a):
 m,_,diag=e.segment(a);return np.clip(e.fullmask(m,a),0,1),diag

def main():
 started=time.monotonic();records=[];panels=[];photos=[]
 paths=[OUT/'assets/portrait.jpg',OUT/'assets/pose.jpg',e.ROOT/'experimental/skin-protection-20261019/reference_color_app/demo/skin-target.png']
 for path in paths:
  a=load(path);m,diag=mask(a);photos.append((a,m))
  # Known mild foreground warm / background cool transform, no tuning after results.
  f=a.astype(float);desired=np.rint(np.clip(m[...,None]*(f*[1.02,.96,.93]+[5,0,0])+(1-m[...,None])*(f*[.87,.99,1.07]+[-3,2,7]),0,255)).astype(np.uint8)
  rm,rd=mask(desired);entry=dict(file=path.name,target_mask=diag,reference_mask=rd,results=[])
  panels.extend([(path.stem+' input',a),(path.stem+' controlled reference',desired)])
  for amount in (0,.5):
   mp=fit(a,desired,m,rm,amount);out=e.apply_mapping(a,mp,.8)
   Image.fromarray(out).save(OUT/(path.stem+f'-amount-{amount}.png'))
   mae=float(np.abs(out.astype(float)-desired).mean())
   saturation_clip=float(((out==0)|(out==255)).any(axis=2).mean())
   entry['results'].append(dict(amount=amount,mae_vs_100_percent_control_reference=mae,any_channel_at_boundary_fraction=saturation_clip))
   panels.append((path.stem+f' amount {amount}',out))
  records.append(entry)
 # Pair different-content photos to show possible skin/clothing/material interference.
 cross=[]
 for i,(a,m) in enumerate(photos):
  b,bm=photos[(i+1)%len(photos)];row=[]
  for amount in (0,.5):
   out=e.apply_mapping(a,fit(a,b,m,bm,amount),.8);row.append(out)
  cross.extend([(f'pair {i} target',a),(f'pair {i} reference',b),(f'pair {i} dense control',row[0]),(f'pair {i} balanced',row[1])])
 e.OUT=OUT;e.grid('controlled-photos.png',panels);e.grid('cross-content.png',cross)
 # Same single global cube applies to the whole clip, no spatial per-frame effect.
 video=e.ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
 first,_=e.media.decode_frame(video,0);fm,_=mask(first);ref,rm=photos[0]
 mp=fit(first,ref,fm,rm,.5);(OUT/'grade.cube').write_text(cube_text(mp,.8),'ascii')
 ff=str(e.media.ffmpeg_path());output=OUT/'global-balanced-video.mp4'
 args=[ff,'-hide_banner','-nostdin','-y','-i',str(video),'-map','0:v:0','-map','0:a:0?',
  '-vf','format=rgb24,lut3d=file=grade.cube:interp=trilinear,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p',
  '-fps_mode','passthrough','-c:v','libx264','-crf','18','-c:a','aac',str(output)]
 subprocess.run(args,cwd=OUT,check=True,capture_output=True,timeout=120)
 verify=OUT/'decode-progress.txt'
 subprocess.run([ff,'-hide_banner','-nostdin','-xerror','-i',str(output),'-map','0:v:0','-an','-progress',str(verify),'-f','null','-'],check=True,capture_output=True,timeout=60)
 frames=int(re.findall(r'^frame=(\d+)\s*$',verify.read_text(),re.M)[-1]);assert frames==75,frames
 metadata=e.media.probe_media(output);assert abs(metadata['duration_seconds']-3)<.1
 report=dict(photos=records,cross_content_pairs=3,video=dict(frames=frames,seconds=metadata['duration_seconds'],mode='One fixed combined global cube; source 25fps preserved',audio_tested=False),
  limits=['portrait and pose are newly added; third photo was already used','Controlled references are generated from source using model masks, not independent style truth',
  'No aesthetic blind review; clipping fraction alone is not a quality verdict','No new App UI or default behavior change; no camera footage benchmark'],
  elapsed_seconds=time.monotonic()-started)
 (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
