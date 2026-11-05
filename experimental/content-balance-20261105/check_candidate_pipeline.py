"""Real file baseline regression, actual PNG/CUBE/video export, no UI change."""
from pathlib import Path
import sys,json,subprocess,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image,ImageDraw
from candidate_pipeline import analyze
from reference_color_app import media,pipeline
from reference_color_app.lut import apply_mapping,cube_text

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/content-balance-20261105/baseline-adapter';OUT.mkdir(exist_ok=True)

def main():
 assets=ROOT/'outputs/content-balance-20261105/person-gate-v4/assets'
 old=ROOT/'outputs/content-balance-20261105/holdout-v3/assets'
 pairs=[(assets/'lena.jpg',assets/'fruits.jpg'),(old/'cat.jpg',assets/'lena.jpg'),
        (assets/'lena.jpg',assets/'lena.jpg'),(assets/'lena.jpg',assets/'messi5.jpg')]
 records=[];sheet=Image.new('RGB',(1280,320*len(pairs)),'#202428');draw=ImageDraw.Draw(sheet)
 for index,(target,reference) in enumerate(pairs):
  baseline=pipeline.analyze(target,reference);disabled=analyze(target,reference,enabled=False)
  candidate=analyze(target,reference,enabled=True);a=media.decode_image(target);b=media.decode_image(reference)
  before=apply_mapping(a,baseline,.8);after=apply_mapping(a,candidate,.8)
  applied=candidate.diagnostics['candidate']['applied']
  assert np.array_equal(baseline.layers,disabled.layers)
  if not applied:assert np.array_equal(baseline.layers,candidate.layers) and np.array_equal(before,after)
  Image.fromarray(after).save(OUT/f'case{index}-candidate.png')
  (OUT/f'case{index}.cube').write_text(cube_text(candidate,.8),'ascii')
  records.append(dict(target=target.name,reference=reference.name,applied=applied,
    disabled_exact=True,fallback_exact=None if applied else True,
    mean_absolute_change_from_original=float(np.abs(before.astype(float)-after).mean()),
    note='Output change is NOT aesthetic improvement',candidate=candidate.diagnostics['candidate']))
  for col,(label,rgb) in enumerate((('target',a),('reference',b),('actual baseline',before),('candidate',after))):
   im=Image.fromarray(rgb);im.thumbnail((310,280));x=col*320;y=index*320
   draw.text((x+6,y+5),f'{index}: {label}',fill='white');sheet.paste(im,(x+(320-im.width)//2,y+30))
 sheet.save(OUT/'actual-baseline-comparison.png')
 video=ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
 ff=str(media.ffmpeg_path());source=OUT/'with-audio.mp4'
 subprocess.run([ff,'-hide_banner','-nostdin','-y','-i',str(video),'-f','lavfi','-i','sine=frequency=440:sample_rate=48000',
  '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-shortest',str(source)],check=True,capture_output=True,timeout=60)
 mapping=analyze(source,assets/'lena.jpg',enabled=True);(OUT/'video.cube').write_text(cube_text(mapping,.8),'ascii')
 output=OUT/'candidate-with-audio.mp4'
 subprocess.run([ff,'-hide_banner','-nostdin','-y','-i',str(source),'-map','0:v:0','-map','0:a:0',
  '-vf','format=rgb24,lut3d=file=video.cube:interp=trilinear,format=yuv420p','-fps_mode','passthrough',
  '-c:v','libx264','-crf','18','-c:a','aac',str(output)],cwd=OUT,check=True,capture_output=True,timeout=120)
 progress=OUT/'decode.txt'
 subprocess.run([ff,'-hide_banner','-nostdin','-xerror','-i',str(output),'-progress',str(progress),'-f','null','-'],
  check=True,capture_output=True,timeout=60)
 frames=int(re.findall(r'^frame=(\d+)\s*$',progress.read_text(),re.M)[-1]);meta=media.probe_media(output)
 assert frames==75 and meta['has_audio'] and abs(meta['duration_seconds']-3)<.1
 report=dict(cases=records,video=dict(frames=frames,has_audio=True,duration=meta['duration_seconds'],
  audio='Synthetic 440Hz tone; decode verified, no listening test',candidate=mapping.diagnostics['candidate']),
  product_changed=False,limits=['No aesthetic improvement claim','No UI integration or portable release','No long-video or HDR test'])
 (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(dict(cases=[{k:v for k,v in r.items() if k!='candidate'} for r in records],video_frames=frames,has_audio=True),indent=2))

if __name__=='__main__':main()
