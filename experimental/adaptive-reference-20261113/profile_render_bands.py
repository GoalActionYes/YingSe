"""Measure existing render band options without changing product defaults."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from reference_color_app.media import decode_frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected single run directory')
    out=ROOT/'outputs/adaptive-reference-20261113/render-performance'/args.run;out.mkdir(parents=True,exist_ok=False)
    for name in ['profile_render_bands.py','spatial_render.py','training_checks.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(4);engine=SpatialEngine('multiref-v1')
    source=ROOT/'outputs/视频兼容副本/P1106511-8bit-SDR.mp4';native,_=decode_frame(source,1.)
    ref_path=ROOT/'outputs/adaptive-reference-20261113/spatial-workbench/acceptance-current-20261114-v1/reference.png'
    with Image.open(ref_path) as image:reference=np.array(image.convert('RGB'))
    report=dict(scope='Single real native decoded frame, resized crops for timing; existing exposed band options; no decoder/encoder/UI timing, no product default changes',
                torch=torch.__version__,gpu=torch.cuda.get_device_name(),checkpoint_sha256=engine.checkpoint_sha,
                source_sha256=digest(source),reference_sha256=digest(ref_path),
                code_sha256={n:digest(out/n) for n in ['profile_render_bands.py','spatial_render.py','training_checks.py']},cases=[],finished=False)
    for width,height in [(640,360),(1920,1080),(2160,3840)]:
        rgb=native if (width,height)==(2160,3840) else np.array(ImageOps.fit(Image.fromarray(native),(width,height),Image.Resampling.BILINEAR))
        prepared=engine.prepare(rgb,reference);gold=prepared.float_output(rgb,128)
        bands=[128,256,512] if width>640 else [128,256,360]
        for band in bands:prepared.apply(rgb,chunk_rows=band)
        rows=[]
        for repeat in range(6):
            for band in (bands if repeat%2==0 else bands[::-1]):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                generated=engine.prepare(rgb,reference);analysis=time.perf_counter()-start
                start=time.perf_counter();raw=generated.float_output(rgb,band);render=time.perf_counter()-start
                error=float(np.abs(raw-gold).max());assert error==0,(width,height,band,error)
                rows.append(dict(repeat=repeat,band=band,analysis_seconds=analysis,render_seconds=render,
                                 combined_seconds=analysis+render,max_float_error=error,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        summary={str(band):{key:float(np.median([r[key] for r in rows if r['band']==band])) for key in ['analysis_seconds','render_seconds','combined_seconds','peak_allocated_bytes']} for band in bands}
        report['cases'].append(dict(width=width,height=height,rows=rows,summary=summary));print(json.dumps(dict(width=width,height=height,summary=summary)),flush=True)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
