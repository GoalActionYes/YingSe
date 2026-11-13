"""Full-length natural motion diagnostic; fixed evaluation ROI never guides grading."""
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
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping


def main():
    torch.set_num_threads(4);engine=SpatialEngine()
    out=ROOT/'outputs/adaptive-reference-20261113/natural-video/motion-v2';out.mkdir(parents=True,exist_ok=False)
    for name in ['check_natural_video.py','natural-video-protocol.md']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    source=ROOT/'outputs/视频兼容副本/P1106511-8bit-SDR.mp4'
    data=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2';records=json.loads((data/'results.json').read_text());item=next(r for r in records['cases'] if r['index']==67849)['triplet']
    ref_path=data/'srgb'/(item['reference']+'.png')
    with Image.open(ref_path) as im:reference=np.array(im.convert('RGB'))
    width,height=720,1280;size=width*height*3;log=(out/'ffmpeg.log').open('wb');flags=subprocess.CREATE_NO_WINDOW;wall_started=time.perf_counter()
    decoder=subprocess.Popen([str(ffmpeg_path()),'-v','error','-nostdin','-i',str(source),'-an','-vf',f'fps=30,scale={width}:{height}','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log,creationflags=flags)
    encoder=subprocess.Popen([str(ffmpeg_path()),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(out/'dynamic.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=flags)
    rows=[];mapping=None;previous=None;roi=(slice(round(height*.02),round(height*.08)),slice(round(width*.70),round(width*.95)))
    try:
        while True:
            raw=bytearray()
            while len(raw)<size:
                chunk=decoder.stdout.read(size-len(raw))
                if not chunk:break
                raw.extend(chunk)
            if not raw:break
            if len(raw)!=size:raise ValueError('Partial natural-video input frame')
            frame=np.frombuffer(raw,np.uint8).reshape(height,width,3)
            if mapping is None:
                first=out/'analysis-frame.png';Image.fromarray(frame).save(first);mapping=analyze(first,ref_path)
            start=time.perf_counter();result=engine.prepare(frame,reference).apply(frame);seconds=time.perf_counter()-start
            legacy_roi=apply_mapping(frame[roi],mapping,1)
            means=dict(input=frame[roi].astype(float).mean((0,1)).tolist(),legacy=legacy_roi.astype(float).mean((0,1)).tolist(),dynamic=result[roi].astype(float).mean((0,1)).tolist())
            row=dict(frame=len(rows),seconds=seconds,roi_means=means)
            if previous is not None:row['roi_mean_step']={name:float(np.abs(np.array(value)-np.array(previous[name])).mean()) for name,value in means.items()}
            rows.append(row);previous=means;encoder.stdin.write(result.tobytes())
            if len(rows) in [1,241,601]:
                Image.fromarray(frame).save(out/f'input-{len(rows)-1}.jpg',quality=95);Image.fromarray(result).save(out/f'output-{len(rows)-1}.jpg',quality=95)
            if len(rows)%90==0:print(json.dumps(dict(frames=len(rows),median_seconds=float(np.median([r['seconds'] for r in rows])))),flush=True)
        encoder.stdin.close();encoder.stdin=None
        if decoder.wait(timeout=30) or encoder.wait(timeout=90):raise ValueError('Natural-video codec failed')
    finally:
        for process in [decoder,encoder]:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            for stream in [process.stdin,process.stdout]:
                if stream is not None:stream.close()
        log.close()
    summary={}
    for name in ['input','legacy','dynamic']:
        values=np.array([r['roi_means'][name] for r in rows]);steps=np.abs(np.diff(values,axis=0)).mean(1)
        summary[name]=dict(channel_std=values.std(0).tolist(),mean_step=float(steps.mean()),p95_step=float(np.quantile(steps,.95)),max_step=float(steps.max()))
    residual=np.array([r['roi_means']['dynamic'] for r in rows])-np.array([r['roi_means']['input'] for r in rows])
    report=dict(scope='Natural person motion at downscaled720x1280; fixed source-observed evaluation ROI, no temporal GT or full 4K/AV acceptance',
        source_sha256=digest(source),reference_sha256=digest(ref_path),script_sha256=digest(Path(__file__)),frames=len(rows),fps=30,
        roi_normalized=[.70,.02,.95,.08],summary=summary,dynamic_minus_input_channel_std=residual.std(0).tolist(),rows=rows,
        wall_seconds=time.perf_counter()-wall_started,legacy_roi_optimization='Exact full-raster/ROI equality verified on same fixed global mapping; ROI only changes evaluation cost',
        output_metadata=probe_media(out/'dynamic.mp4'),output_sha256=digest(out/'dynamic.mp4'),product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(frames=len(rows),summary=summary)),flush=True)


if __name__=='__main__':main()
