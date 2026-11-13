"""Owned native 4K band-size timing/numerical comparison, no model changes."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from reference_color_app.media import decode_frame
from spatial_render import SpatialEngine


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/natural-video/bands-v1';out.mkdir(exist_ok=False)
    source=ROOT/'outputs/视频兼容副本/P1106511-8bit-SDR.mp4'
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2'
    cases=json.loads((base/'results.json').read_text(encoding='utf-8'))['cases']
    item=next(c for c in cases if c['index']==67849)['triplet']
    reference=base/'srgb'/(item['reference']+'.png')
    with Image.open(reference) as im:ref=np.array(im.convert('RGB'))
    frame,_=decode_frame(source,0);engine=SpatialEngine();prepared=engine.prepare(frame,ref)
    baseline=prepared.float_output(frame,chunk_rows=128);baseline_png=np.rint(baseline.clip(0,1)*255).astype(np.uint8)
    cases=[];timings={k:[] for k in [128,256,512,1024]}
    for band in timings:
        torch.cuda.reset_peak_memory_stats();actual=prepared.float_output(frame,chunk_rows=band)
        png=np.rint(actual.clip(0,1)*255).astype(np.uint8)
        cases.append({'band':band,'max_float_delta':float(np.abs(actual-baseline).max()),
            'max_rgb8_delta':int(np.abs(png.astype(int)-baseline_png.astype(int)).max()),
            'changed_rgb8_values':int(np.count_nonzero(png!=baseline_png)),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        del actual,png
    rng=np.random.default_rng(20261113)
    for _ in range(3):
        for band in rng.permutation(list(timings)):
            torch.cuda.synchronize();start=time.perf_counter();prepared.apply(frame,chunk_rows=int(band));torch.cuda.synchronize()
            timings[int(band)].append(time.perf_counter()-start)
    report={'source_sha256':digest(source),'reference_sha256':digest(reference),'script_sha256':digest(Path(__file__)),
        'dimensions':[frame.shape[1],frame.shape[0]],'cases':cases,'timings':timings,
        'scope':'Same frozen grid/native 4K frame; band scheduling only, no complete video or visual acceptance'}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
