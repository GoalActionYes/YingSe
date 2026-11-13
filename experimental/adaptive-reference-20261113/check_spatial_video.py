"""Known-pan diagnostic and a bounded actual-person-video export smoke test."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from reference_color_app.media import ffmpeg_path,probe_media


def main():
    torch.set_num_threads(4);engine=SpatialEngine()
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-render/video-v1';out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
    meta=probe_media(source)
    if (meta['width'],meta['height'])!=(640,360) or meta['hdr_unsupported'] or meta['has_audio']:raise ValueError('Unexpected fixed smoke clip')
    data=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2';record=json.loads((data/'results.json').read_text(encoding='utf-8'))
    item=next(r for r in record['cases'] if r['index']==67849)['triplet']
    with Image.open(data/'srgb'/(item['reference']+'.png')) as im:reference=np.array(im.convert('RGB'))
    with Image.open(data/'srgb'/(item['content']+'.png')) as im:photo=np.array(im.convert('RGB').resize((640,360),Image.Resampling.BILINEAR))
    initial=engine.prepare(photo,reference);first=initial.apply(photo);pan=[];strength_jump=[]
    for index in range(24):
        shift=index*3;frame=np.roll(photo,shift,axis=1);dynamic=engine.prepare(frame,reference).apply(frame)
        aligned_reference=np.roll(first,shift,axis=1)
        error=np.abs(dynamic[:,80:-80].astype(float)-aligned_reference[:,80:-80].astype(float))
        pan.append(dict(frame=index,shift=shift,aligned_mae=float(error.mean()),aligned_p95=float(np.quantile(error,.95))))
    # Deliberate tiny illumination changes: this is a response check, not a GT.
    illumination=[];previous=None
    for index in range(24):
        gain=1+.02*np.sin(index/23*2*np.pi);frame=np.rint(np.clip(photo.astype(float)*gain,0,255)).astype(np.uint8)
        result=engine.prepare(frame,reference).apply(frame)
        if previous is not None:illumination.append(dict(frame=index,input_change=float(np.abs(frame.astype(float)-previous[0]).mean()),output_change=float(np.abs(result.astype(float)-previous[1]).mean())))
        previous=(frame,result)
    raw=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(source),'-t','3','-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdin=subprocess.DEVNULL,capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    if raw.returncode:raise ValueError(raw.stderr.decode('utf-8','replace'))
    if len(raw.stdout)%(640*360*3):raise ValueError('Partial decoded RGB frame')
    frames=np.frombuffer(raw.stdout,np.uint8).reshape(-1,360,640,3)
    if len(frames)!=75:raise ValueError('Unexpected fixed input frame count')
    outputs=[];timings=[]
    for index,frame in enumerate(frames):
        start=time.perf_counter();prepared=engine.prepare(frame,reference);result=prepared.apply(frame);timings.append(time.perf_counter()-start);outputs.append(result)
        if index in [0,37,74]:Image.fromarray(result).save(out/f'frame-{index}.png')
    repeat=engine.prepare(frames[0],reference).apply(frames[0]);repeat_max=int(np.abs(repeat.astype(int)-outputs[0].astype(int)).max())
    destination=out/'person-motion.mp4'
    encode=subprocess.run([str(ffmpeg_path()),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','25','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(destination)],
        input=np.stack(outputs).tobytes(),capture_output=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    if encode.returncode:raise ValueError(encode.stderr.decode('utf-8','replace'))
    decoded=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(destination),'-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdin=subprocess.DEVNULL,capture_output=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    if decoded.returncode or len(decoded.stdout)%(640*360*3):raise ValueError('Cannot verify exported frames')
    decoded_frames=len(decoded.stdout)//(640*360*3)
    report=dict(scope='Known motion/illumination diagnostics and 3s real-video smoke; not full natural-video quality or temporal acceptance',
        source_sha256={name:digest(Path(__file__).with_name(name)) for name in ['spatial_render.py','check_spatial_video.py']},
        input_sha256=digest(source),reference_sha256=digest(data/'srgb'/(item['reference']+'.png')),input_metadata=meta,
        known_pan=pan,known_pan_mean=float(np.mean([p['aligned_mae'] for p in pan])),illumination=illumination,
        illumination_policy='Raw frame difference includes intended brightness changes; not an optical-flow flicker metric',
        input_frames=75,output_decoded_frames=decoded_frames,repeat_frame_max_difference=repeat_max,
        processing_seconds=dict(total=float(sum(timings)),median=float(np.median(timings)),max=float(max(timings))),
        output_metadata=probe_media(destination),output_sha256=digest(destination),
        smoke_passed=decoded_frames==75 and repeat_max<=1,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_metadata','output_metadata','known_pan','illumination']}),flush=True)
    if not report['smoke_passed']:raise SystemExit(1)


if __name__=='__main__':main()
