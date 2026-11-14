"""Fixed same-frame timing/parity probe, no product modification."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from frame_transfer import render


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['app8','multiref-v1'],default='multiref-v1');args=parser.parse_args()
    if 'from frame_transfer import render' in Path(__file__).with_name('spatial_render.py').read_text(encoding='utf-8'):
        raise ValueError('Standalone stage9 probe needs its preintegration frozen-source snapshot; use verify_frame_transfer.py for integrated checks')
    out=ROOT/'outputs/adaptive-reference-20261113/render-performance'/f'frame-transfer-20261115-{args.profile}-v3';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_frame_transfer.py','frame_transfer.py','frame-transfer-protocol.md','test_frame_transfer.py','spatial_render.py','training_checks.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(4);engine=SpatialEngine(args.profile)
    report=dict(scope='Same consumed six images; standalone prepared-render timing excludes analysis/codec/UI, no integration',stage9_band_rows=128,candidate_band_rows=256,profile=args.profile,gpu=torch.cuda.get_device_name(),torch=torch.__version__,checkpoint_sha256=engine.checkpoint_sha,source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False)
    data=ROOT/'outputs/adaptive-reference-20261113/product-styles/current13-20261114-v1'
    for index,size in [(80441,(256,256)),(32004,(640,360)),(37184,(1920,1080)),(67770,(1917,1081)),(46837,(1080,1920)),(64590,(2160,3840))]:
        source=data/str(index)/'source.png';ref=data/str(index)/'cool-reference.png'
        with Image.open(source) as im:rgb=np.array(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))
        with Image.open(ref) as im:reference=np.array(im.convert('RGB'))
        start=time.perf_counter();prepared=engine.prepare(rgb,reference);analysis=time.perf_counter()-start
        old=prepared.byte_output(rgb);candidate=render(prepared,rgb)
        if not np.array_equal(old,candidate):raise ValueError(f'Pixel parity failed at {size}')
        rows=[]
        for repeat in range(6):
            for method in (['stage9','frame'] if repeat%2==0 else ['frame','stage9']):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                actual=prepared.byte_output(rgb) if method=='stage9' else render(prepared,rgb)
                seconds=time.perf_counter()-start
                if not np.array_equal(actual,old):raise ValueError('Repeated pixel parity failed')
                rows.append(dict(repeat=repeat,method=method,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        summary={name:{key:float(np.median([r[key] for r in rows if r['method']==name])) for key in ['seconds','peak_allocated_bytes']} for name in ['stage9','frame']}
        report['cases'].append(dict(index=index,size=list(size),source_sha256=digest(source),reference_sha256=digest(ref),max_rgb8_difference=0,analysis_seconds=analysis,timings=rows,summary=summary))
        print(json.dumps(dict(size=size,summary=summary)),flush=True)
    lookup={tuple(r['size']):r for r in report['cases']}
    passed=all(lookup[s]['summary']['frame']['seconds']<=.85*lookup[s]['summary']['stage9']['seconds'] for s in [(1920,1080),(1080,1920)]) and lookup[(2160,3840)]['summary']['frame']['seconds']<=1.05*lookup[(2160,3840)]['summary']['stage9']['seconds'] and max(t['peak_allocated_bytes'] for c in report['cases'] for t in c['timings'])<2*1024**3
    report.update(finished=True,passed_performance_gate=passed,product_integrated=False)
    if any(digest(Path(__file__).with_name(n))!=v for n,v in report['source_sha256'].items()):raise ValueError('Source changed')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(profile=args.profile,passed=passed)),flush=True)


if __name__=='__main__':main()
