"""Unintegrated RGB8 transfer experiment, compared against current float CPU oracle."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine,coordinates,validate_rgb
from training_checks import finite_output


@torch.inference_mode()
def byte_candidate(prepared,rgb,chunk_rows=256):
    validate_rgb(rgb)
    if type(chunk_rows)!=int or chunk_rows<1:raise ValueError('Positive band height required')
    engine=prepared.engine;model=engine.model;h,w=rgb.shape[:2]
    output=np.empty(rgb.shape,np.uint8);weights=F.softmax(model.sampler.L_weights,dim=1)
    for start in range(0,h,chunk_rows):
        stop=min(h,start+chunk_rows);c=engine.tensor(rgb[start:stop]);z=(model.sampler.pre_guide(c)*weights).sum(1)
        finite_output(z,(1,stop-start,w))
        coeff=F.grid_sample(prepared.grid,coordinates(w,h,start,stop,z),mode='bilinear',align_corners=False,padding_mode='border').squeeze(2)
        raw=finite_output(model.apply_coeffs(coeff,c),c.shape)
        output[start:stop]=raw[0].clamp(0,1).mul(255).round().to(torch.uint8).cpu().permute(1,2,0).numpy()
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--integrated',action='store_true');parser.add_argument('--profile',choices=['app8','multiref-v1'],default='multiref-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected one run name')
    out=ROOT/'outputs/adaptive-reference-20261113/render-performance'/args.run;out.mkdir(parents=True,exist_ok=False)
    for name in ['probe_byte_render.py','spatial_render.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(4);engine=SpatialEngine(args.profile);report=dict(profile=args.profile,checkpoint_sha256=engine.checkpoint_sha,source_sha256={p.name:digest(p) for p in out.iterdir()},scope='Unintegrated full-strength RGB8 quantization/transfer experiment, no product changes',cases=[],finished=False)
    data=ROOT/'outputs/adaptive-reference-20261113/product-styles/current13-20261114-v1'
    for index,size in [(80441,(256,256)),(32004,(640,360)),(37184,(1920,1080)),(67770,(1917,1081)),(46837,(1080,1920)),(64590,(2160,3840))]:
        with Image.open(data/str(index)/'source.png') as im:rgb=np.array(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))
        with Image.open(data/str(index)/'cool-reference.png') as im:reference=np.array(im.convert('RGB'))
        prepared=engine.prepare(rgb,reference)
        def old_full():
            raw=prepared.float_output(rgb,128);blended=rgb.astype(np.float32)/255*0+raw*1
            return np.rint(np.clip(blended,0,1)*255).astype(np.uint8)
        def current():return prepared.apply(rgb,1) if args.integrated else byte_candidate(prepared,rgb)
        gold=old_full()
        candidate=current();delta=np.abs(gold.astype(int)-candidate.astype(int));maximum=int(delta.max());count=int(np.count_nonzero(delta))
        if maximum:raise AssertionError((index,size,maximum,count))
        timings=[]
        for repeat in range(6):
            for name in (['old','byte'] if repeat%2==0 else ['byte','old']):
                torch.cuda.synchronize();start=time.perf_counter()
                value=old_full() if name=='old' else current()
                timings.append(dict(repeat=repeat,method=name,seconds=time.perf_counter()-start));assert np.array_equal(value,gold)
        row=dict(index=index,size=list(size),max_rgb_error=maximum,changed_values=count,timings=timings,medians={k:float(np.median([r['seconds'] for r in timings if r['method']==k])) for k in ['old','byte']})
        report['cases'].append(row);print(json.dumps(row),flush=True)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['integrated']=args.integrated;report['finished']=True
    if args.integrated:report['scope']='Integrated full-strength RGB8 renderer vs original float/CPU formula, same128-row bands; excludes prepare/codec/UI'
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
