"""Author full-raster oracle, seams, strength and bounded native rendering."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from training_checks import finite_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--multiref',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-render'/('native-multiref-v1' if args.multiref else 'native-v1');out.mkdir(parents=True,exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['spatial_render.py','check_spatial_render.py','spatial-render-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    engine=SpatialEngine('multiref-v1' if args.multiref else 'app8');rng=np.random.default_rng(384);rows=[]
    data=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2'
    records=json.loads((data/'results.json').read_text(encoding='utf-8'));item=next(r for r in records['cases'] if r['index']==67849)['triplet']
    def load(relative):
        with Image.open(data/'srgb'/(relative+'.png')) as im:return im.convert('RGB').copy()
    photo,ref=load(item['content']),load(item['reference']);reference=np.array(ref)
    targets=[('noise256',rng.integers(0,256,(256,256,3),dtype=np.uint8)),('photo_rect',np.array(photo.resize((431,279),Image.Resampling.BILINEAR))),('extremes',np.repeat(np.linspace(0,255,337,dtype=np.uint8)[None,:,None],211,0).repeat(3,2))]
    for name,target in targets:
        prepared=engine.prepare(target,reference);c=engine.tensor(target)
        with torch.inference_mode():
            coeff=engine.model.sampler(prepared.grid,c);raw=finite_output(engine.model.apply_coeffs(coeff,c),c.shape)
            oracle=raw[0].cpu().permute(1,2,0).numpy()
            whole=prepared.float_output(target,chunk_rows=target.shape[0]);band17=prepared.float_output(target,chunk_rows=17);band128=prepared.float_output(target,chunk_rows=128)
            errors={key:float(np.abs(value-oracle).max()) for key,value in [('whole',whole),('band17',band17),('band128',band128)]}
            direct_error=None
            if name=='noise256':
                s=engine.analysis(reference);hc=engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](c));hs=engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](s))
                direct=engine.model(c,hc,hs)[0][0].cpu().permute(1,2,0).numpy();direct_error=float(np.abs(direct-band17).max())
        zero=prepared.apply(target,0);half=prepared.apply(target,.5);full=prepared.apply(target,1)
        expected_half=np.rint(np.clip(target.astype(np.float32)/255*.5+band128*.5,0,1)*255).astype(np.uint8)
        quantized=np.rint(np.clip(oracle,0,1)*255).astype(np.uint8)
        rows.append(dict(name=name,shape=list(target.shape),max_float_error=errors,direct_forward_max_error=direct_error,
            zero_exact=bool(np.array_equal(zero,target)),half_exact=bool(np.array_equal(half,expected_half)),quantized_max_error=int(np.abs(full.astype(int)-quantized.astype(int)).max())))
        Image.fromarray(full).save(out/f'{name}.png')
    timing=[]
    for width,height in [(1920,1080),(3840,2160)]:
        target=np.array(photo.resize((width,height),Image.Resampling.BILINEAR));prepared=engine.prepare(target,reference)
        samples=[]
        for repeat in range(2):
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter();result=prepared.apply(target,1,chunk_rows=128);seconds=time.perf_counter()-start
            samples.append(dict(seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        path=out/f'{width}x{height}.png';Image.fromarray(result).save(path)
        timing.append(dict(dimensions=[width,height],analysis_seconds=prepared.diagnostics['analysis_seconds'],renders=samples,sha256=digest(path),source_is_upscaled=True))
    passed=all(max(r['max_float_error'].values())<2e-5 and (r['direct_forward_max_error'] is None or r['direct_forward_max_error']<2e-5) and r['zero_exact'] and r['half_exact'] and r['quantized_max_error']<=1 for r in rows)
    report=dict(scope='Numerical native renderer and upscaled-photo speed, not photographic/video/product acceptance',
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},checkpoint_sha256=prepared.diagnostics['checkpoint_sha256'],
        numerical=rows,timing=timing,float_max_gate=2e-5,quantized_max_gate=1,passed=passed,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
