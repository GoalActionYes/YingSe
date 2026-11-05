"""Frozen visible-face eligibility validation, including explicit false negatives."""
from pathlib import Path
import sys,json,time,subprocess,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
import evaluate as e
from check_natural import load
from composition import extract_pair
from dense_features import extract_dense
from person_gate import segment,fullmask,VERSION
from reference_color_app.lut import cube_text

OUT=e.ROOT/'outputs/content-balance-20261105/person-gate-v4'

def fit(a,b,am,bm):
 x,y,d=extract_pair(a,b,am,bm,.5)
 return e.build_mapping(x,y),d

def main():
 OUT.mkdir(exist_ok=True);(OUT/'masks').mkdir(exist_ok=True);started=time.monotonic();rows=[];panels=[];loaded=[]
 paths=[(p,p.name!='cat.jpg','observed-regression') for p in (e.ROOT/'outputs/content-balance-20261105/holdout-v3/assets').glob('*')]
 paths += [(p,True,'observed-regression') for p in (e.ROOT/'outputs/content-balance-20261105/natural-check/assets').glob('*.jpg')]
 paths += [(e.ROOT/'experimental/skin-protection-20261019/reference_color_app/demo/skin-target.png',True,'observed-regression')]
 sources=json.loads((OUT/'sources.json').read_text('utf8'))
 for s in sources:
  if 'error' not in s:paths.append((OUT/'assets'/s['file'],s['file'] in ('lena.jpg','messi5.jpg','basketball1.png'),'new-fixed-validation'))
 for path,person,split in paths:
  a=load(path);old,_,od=e.segment(a);new,_,nd=segment(a)
  record=dict(file=path.name,expected_contains_person=person,split=split,old=od,new=nd,
   accepted_masks_exactly_unchanged=bool(np.array_equal(old,new)) if nd['person_detected'] else None)
  rows.append(record);loaded.append((a,fullmask(new,a),record))
  oldfull=fullmask(old,a);newfull=fullmask(new,a)
  Image.fromarray(np.rint(oldfull*255).astype(np.uint8)).save(OUT/'masks'/(path.stem+'-raw.png'))
  Image.fromarray(np.rint(newfull*255).astype(np.uint8)).save(OUT/'masks'/(path.stem+'-gated.png'))
  panels.extend([(path.name+' input',a),(path.name+' raw people mask',np.repeat(np.rint(oldfull*255).astype(np.uint8)[...,None],3,axis=2)),
    (path.name+(' ALLOW' if nd['person_detected'] else 'FALL BACK'),np.repeat(np.rint(newfull*255).astype(np.uint8)[...,None],3,axis=2))])
 e.OUT=OUT;e.grid('mask-validation.png',panels)
 anchor=next(v for v in loaded if v[2]['file']=='portrait.jpg');fallback_checks=[]
 for a,am,row in loaded:
  if row['new']['person_detected']:continue
  for direction in ('rejected-target','rejected-reference'):
   target,reference,tm,rm=(a,anchor[0],am,anchor[1]) if direction=='rejected-target' else (anchor[0],a,anchor[1],am)
   try:
    mapping,diag=fit(target,reference,tm,rm)
    baseline=e.build_mapping(extract_dense(target,0),extract_dense(reference,1))
    same_layers=bool(np.array_equal(mapping.layers,baseline.layers))
    same_output=bool(np.array_equal(e.apply_mapping(target,mapping,.8),e.apply_mapping(target,baseline,.8)))
    fallback_checks.append(dict(file=row['file'],direction=direction,identical_lut=same_layers,identical_output=same_output,adaptation_applied=diag['applied']))
   except ArithmeticError as ex:fallback_checks.append(dict(file=row['file'],direction=direction,error=str(ex)))
 # Exercise the first-frame analysis -> fixed global map -> video path.
 video=e.ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
 first,_=e.media.decode_frame(video,0);vm,_,vd=segment(first)
 mapping,diag=fit(first,anchor[0],fullmask(vm,first),anchor[1])
 (OUT/'grade.cube').write_text(cube_text(mapping,.8),'ascii')
 ff=str(e.media.ffmpeg_path());output=OUT/'face-gated-video.mp4'
 subprocess.run([ff,'-hide_banner','-nostdin','-y','-i',str(video),'-map','0:v:0','-map','0:a:0?',
  '-vf','format=rgb24,lut3d=file=grade.cube:interp=trilinear,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p',
  '-fps_mode','passthrough','-c:v','libx264','-crf','18','-c:a','aac',str(output)],cwd=OUT,check=True,capture_output=True,timeout=120)
 progress=OUT/'decode-progress.txt'
 subprocess.run([ff,'-hide_banner','-nostdin','-xerror','-i',str(output),'-map','0:v:0','-an','-progress',str(progress),'-f','null','-'],check=True,capture_output=True,timeout=60)
 frames=int(re.findall(r'^frame=(\d+)\s*$',progress.read_text(),re.M)[-1]);assert frames==75
 false_accepts=[r['file'] for r in rows if not r['expected_contains_person'] and r['new']['person_detected']]
 false_rejects=[r['file'] for r in rows if r['expected_contains_person'] and not r['new']['person_detected']]
 report=dict(version=VERSION,research_only=True,product_changed=False,records=rows,fallback_checks=fallback_checks,
  false_accepts=false_accepts,person_images_that_fell_back=false_rejects,
  accepted_masks_unchanged=all(r['accepted_masks_exactly_unchanged'] is not False for r in rows),
  all_fallback_checks_exact=all(r.get('identical_lut') and r.get('identical_output') and not r.get('adaptation_applied') for r in fallback_checks),
  video=dict(frames=frames,duration=e.media.probe_media(output)['duration_seconds'],first_frame=vd,adaptation=diag,audio_tested=False),
  limits=['Visible-face eligibility is NOT an independently trained person detector','No threshold tuning on new-fixed-validation images',
   'Tiny/occluded/back-facing/poorly segmented people may intentionally fall back','No aesthetic approval, long video or audio validation'],elapsed_seconds=time.monotonic()-started)
 (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(dict(false_accepts=false_accepts,person_images_that_fell_back=false_rejects,
  exact_fallback=report['all_fallback_checks_exact'],accepted_masks_unchanged=report['accepted_masks_unchanged'],
  video_frames=frames,rows=[dict(file=r['file'],person=r['expected_contains_person'],allowed=r['new']['person_detected'],face=r['new']['face_evidence']) for r in rows]),indent=2))

if __name__=='__main__':main()
