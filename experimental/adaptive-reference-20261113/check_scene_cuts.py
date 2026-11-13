"""Frozen real-material cuts: evaluate cross-shot color carry, not style GT."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from temporal_grid import TemporalGrid
from reference_color_app.media import ffmpeg_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='baseline-v1');parser.add_argument('--low-contrast',action='store_true');parser.add_argument('--relative-cuts',action='store_true');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected directory name')
    torch.set_num_threads(4);base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    out=ROOT/'outputs/adaptive-reference-20261113/scene-cuts'/args.run;out.mkdir(parents=True,exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_scene_cuts.py','scene-cut-protocol.md','temporal_grid.py','spatial_render.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sources=[(ROOT/'outputs/视频兼容副本/P1106511-8bit-SDR.mp4','1f4350aabf4382a0c4325a6439dfe984a6d6f3a965a6bec08f26ed5d08c10eec'),
        (ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4','3687fd072b64bce70f3855c750efb0b219aaa9be5f59745a4927ad3f9a0847d9')]
    cohort_path=base/'cohort-multiref-v1/cohort.json';prep_path=base/'cohort-multiref-v1/srgb-v1/manifest.json'
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'));lookup={r['original_path']:r for r in prep['files']}
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path):raise ValueError('Changed development data')
    selected=cohort['groups']['validation'][:8]
    if len({r['content_family'] for r in selected})!=8 or any(r['style_index']!=0 for r in selected):raise ValueError('Changed fixed photo selection')
    records=json.loads((base/'reserved-v2/results.json').read_text(encoding='utf-8'));reference_row=next(r['triplet'] for r in records['cases'] if r['index']==67849)
    ref_path=base/'reserved-v2/srgb'/(reference_row['reference']+'.png')
    report=dict(scope='Controlled cuts of already used natural materials, not new independent style/semantic/AV acceptance',
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},video_sha256=[sha for _,sha in sources],reference_sha256=digest(ref_path),
        cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),selected_photo_indices=[r['index'] for r in selected],
        dimensions=[360,640],fps=30,boundaries=[30,60],low_contrast=args.low_contrast,contrast_factor=.35 if args.low_contrast else 1.,temporal_parameters=TemporalGrid(relative_cut=args.relative_cuts).settings(),limits=dict(mean_code=1.,p95_code=3.),finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    frames=[]
    for path,sha in sources:
        if digest(path)!=sha:raise ValueError('Changed fixed video')
        result=subprocess.run([str(ffmpeg_path()),'-v','error','-nostdin','-i',str(path),'-t','3','-an','-vf','fps=30,scale=360:640:force_original_aspect_ratio=increase,crop=360:640','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdin=subprocess.DEVNULL,capture_output=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode or len(result.stdout)!=90*360*640*3:raise ValueError('Incomplete fixed video decode')
        decoded=np.frombuffer(result.stdout,np.uint8).reshape(90,640,360,3)
        if args.low_contrast:decoded=np.rint(127.5+.35*(decoded.astype(np.float32)-127.5)).astype(np.uint8)
        frames.append(decoded)
    with Image.open(ref_path) as image:reference=np.array(image.convert('RGB'))
    engine=SpatialEngine('multiref-v1');report['checkpoint_sha256']=engine.checkpoint_sha
    def difference(actual,expected):
        delta=np.abs(actual.astype(float)-expected.astype(float));mean=float(delta.mean());p95=float(np.quantile(delta,.95))
        return dict(mean=mean,p95=p95,max=int(delta.max()),passed=mean<=1 and p95<=3)
    def panel(name,source,expected,actual):
        canvas=Image.new('RGB',(1080,665),'#222');draw=ImageDraw.Draw(canvas)
        for index,(label,array) in enumerate([('Input',source),('Fresh current frame',expected),('Temporal output',actual)]):
            draw.text((index*360+5,5),label,fill='white');canvas.paste(Image.fromarray(array),(index*360,25))
        canvas.save(out/(name+'.jpg'),quality=95)
    temporal=TemporalGrid(relative_cut=args.relative_cuts);sequence=list(frames[0][:30])+list(frames[1][:30])+list(frames[0][30:60]);cut_rows=[];resets=[];outputs=[];start=time.perf_counter()
    for index,frame in enumerate(sequence):
        prepared=engine.prepare(frame,reference);smoothed=temporal.smooth(prepared,frame);actual=smoothed.apply(frame);outputs.append(actual)
        reason=smoothed.diagnostics['temporal_reset_reason']
        if reason:resets.append(dict(frame=index,reason=reason))
        if index in [30,60]:
            expected=prepared.apply(frame);row=dict(frame=index,reset=reason,scene_difference=smoothed.diagnostics['temporal_scene_difference'],**difference(actual,expected));cut_rows.append(row)
            panel(f'video-cut-{index}',frame,expected,actual);print(json.dumps(dict(video_boundary=row)),flush=True)
    controls=[]
    for index,video in enumerate(frames):
        temporal.reset();events=[]
        for number,frame in enumerate(video):
            prepared=temporal.smooth(engine.prepare(frame,reference),frame)
            if prepared.diagnostics['temporal_reset_reason']:events.append(dict(frame=number,reason=prepared.diagnostics['temporal_reset_reason']))
        controls.append(dict(source=index,frames=90,resets=events))
    photos=[]
    for row in selected:
        entry=lookup[row['content']];path=base/'cohort-multiref-v1'/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed prepared photo')
        with Image.open(path) as image:photo=np.array(ImageOps.fit(image.convert('RGB'),(360,640),Image.Resampling.BILINEAR))
        if args.low_contrast:photo=np.rint(127.5+.35*(photo.astype(np.float32)-127.5)).astype(np.uint8)
        photos.append((photo,engine.prepare(photo,reference)))
    photo_rows=[]
    for index,(frame,prepared) in enumerate(photos):
        temporal.reset()
        for _ in range(12):temporal.smooth(prepared,frame)
        other,other_prepared=photos[(index+1)%8];smoothed=temporal.smooth(other_prepared,other);actual=smoothed.apply(other);expected=other_prepared.apply(other)
        row=dict(from_index=selected[index]['index'],to_index=selected[(index+1)%8]['index'],reset=smoothed.diagnostics['temporal_reset_reason'],scene_difference=smoothed.diagnostics['temporal_scene_difference'],**difference(actual,expected));photo_rows.append(row)
        panel(f'photo-cut-{index}',other,expected,actual);print(json.dumps(dict(photo_boundary=row)),flush=True)
    path=out/'controlled-cuts.mp4'
    encoded=subprocess.run([str(ffmpeg_path()),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','360x640','-r','30','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],input=np.stack(outputs).tobytes(),capture_output=True,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if encoded.returncode:raise ValueError('Diagnostic encoding failed')
    assert all(digest(path)==sha for path,sha in sources)
    report.update(video_boundaries=cut_rows,video_resets=resets,motion_controls=controls,photo_boundaries=photo_rows,wall_seconds=time.perf_counter()-start,
        finished=True,passed=all(r['passed'] for r in cut_rows+photo_rows),diagnostic_video_sha256=digest(path),product_changed=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(passed=report['passed'],boundaries=10,failed=sum(not r['passed'] for r in cut_rows+photo_rows))),flush=True)


if __name__=='__main__':main()
