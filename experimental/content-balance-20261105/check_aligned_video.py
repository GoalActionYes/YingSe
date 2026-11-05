"""Product pipeline -> CUBE -> actual FFmpeg video with synthetic audio."""
from pathlib import Path
import sys,json,subprocess,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from reference_color_app import pipeline,media
from reference_color_app.lut import apply_mapping,cube_text

OUT=ROOT/'outputs/content-balance-20261105/skin-external-audit/aligned-trial'
def warm(a):return np.rint(np.clip(a.astype(float)*[1.04,.98,.92]+[3,1,0],0,255)).astype(np.uint8)
def main():
 source=ROOT/'outputs/content-balance-20261105/baseline-adapter/with-audio.mp4';first,_=media.decode_frame(source,0)
 ref=OUT/'video-reference.png';Image.fromarray(warm(first)).save(ref)
 mapping=pipeline.analyze(source,ref,0);assert mapping.diagnostics['aligned_reference']['accepted']
 frame_errors=[]
 for t in (0,1,2):
  a,_=media.decode_frame(source,t);out=apply_mapping(a,mapping,1);expected=warm(a)
  err=float(np.abs(out.astype(float)-expected).mean());assert err<.6;frame_errors.append(dict(seconds=t,mean_rgb_error=err))
 (OUT/'video.cube').write_text(cube_text(mapping,1),'ascii');ff=str(media.ffmpeg_path());output=OUT/'aligned-video.mp4'
 def run(args):subprocess.run([ff,'-hide_banner','-nostdin','-y',*args],cwd=OUT,check=True,capture_output=True,timeout=150)
 run(['-i',str(source),'-map','0:v:0','-map','0:a:0','-vf','format=rgb24,lut3d=file=video.cube:interp=trilinear,format=yuv420p',
  '-fps_mode','passthrough','-c:v','libx264','-crf','18','-c:a','aac',str(output)])
 progress=OUT/'video-decode.txt';run(['-xerror','-i',str(output),'-progress',str(progress),'-f','null','-'])
 frames=int(re.findall(r'^frame=(\d+)\s*$',progress.read_text(),re.M)[-1]);meta=media.probe_media(output)
 assert frames==75 and meta['has_audio']
 report=dict(frames=frames,seconds=meta['duration_seconds'],has_audio=True,frame_errors=frame_errors,
  fit=mapping.diagnostics['aligned_reference'],limits=['Known global warm transform, one short clip','Pre-encode error; lossy output may differ','Synthetic audio, no listening test','Not general shot correspondence or different-content matching'])
 (OUT/'video-results.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
