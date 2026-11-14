"""Integrated output/timing/fault checks against hash-verified stage9 byte code."""
import argparse,ast,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine,coordinates,validate_rgb
from training_checks import finite_output


def stage9(profile):
    prior=ROOT/'outputs/adaptive-reference-20261113/render-performance'/f'frame-transfer-20261115-{profile}-v3'
    report=json.loads((prior/'results.json').read_text(encoding='utf-8'));source=prior/'frozen-source/spatial_render.py'
    if digest(source)!=report['source_sha256']['spatial_render.py']:raise ValueError('Original stage9 snapshot changed')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PreparedSpatial')
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='byte_output')
    if 'from frame_transfer import render' in ast.unparse(function):raise ValueError('Snapshot is no longer stage9')
    api=dict(torch=torch,np=np,F=F,coordinates=coordinates,validate_rgb=validate_rgb,finite_output=finite_output)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),api)
    return api['byte_output'],digest(source)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['app8','multiref-v1'],default='multiref-v1');args=parser.parse_args()
    torch.set_num_threads(4);old,oldsha=stage9(args.profile);engine=SpatialEngine(args.profile)
    out=ROOT/'outputs/adaptive-reference-20261113/render-performance'/f'frame-transfer-integrated-20261115-{args.profile}-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['verify_frame_transfer.py','frame_transfer.py','spatial_render.py','spatial_store.py','spatial_build.py','training_checks.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Integrated actualrenderer vs frozenstage9, strengthparity vs128rowfloat; no codec/UI/semantic acceptance',profile=args.profile,checkpoint_sha256=engine.checkpoint_sha,stage9_source_sha256=oldsha,source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[])
    data=ROOT/'outputs/adaptive-reference-20261113/product-styles/current13-20261114-v1'
    fault_fixture=None
    for index,size in [(80441,(256,256)),(32004,(640,360)),(37184,(1920,1080)),(67770,(1917,1081)),(46837,(1080,1920)),(64590,(2160,3840))]:
        with Image.open(data/str(index)/'source.png') as image:rgb=np.array(image.convert('RGB').resize(size,Image.Resampling.BILINEAR))
        with Image.open(data/str(index)/'cool-reference.png') as image:ref=np.array(image.convert('RGB'))
        prepared=engine.prepare(rgb,ref);gold=old(prepared,rgb);raw=prepared.float_output(rgb,128);strengths=[]
        for strength in [0.,.37,1.]:
            expected=np.rint(np.clip(rgb.astype(np.float32)/255*(1-strength)+raw*strength,0,1)*255).astype(np.uint8)
            actual=prepared.apply(rgb,strength)
            if not np.array_equal(actual,expected):raise ValueError(f'Integrated strength parity failed {size} {strength}')
            strengths.append(dict(strength=strength,max_rgb8_difference=0))
        rows=[]
        prepared.apply(rgb,1);old(prepared,rgb)
        for repeat in range(6):
            for name in (['stage9','integrated'] if repeat%2==0 else ['integrated','stage9']):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                actual=old(prepared,rgb) if name=='stage9' else prepared.apply(rgb,1)
                seconds=time.perf_counter()-start
                if not np.array_equal(actual,gold):raise ValueError('Repeated integrated pixels changed')
                rows.append(dict(repeat=repeat,method=name,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        summary={name:float(np.median([r['seconds'] for r in rows if r['method']==name])) for name in ['stage9','integrated']}
        report['cases'].append(dict(index=index,size=list(size),strengths=strengths,timings=rows,medians=summary));print(json.dumps(dict(size=size,medians=summary)),flush=True)
        if size==(640,360):fault_fixture=(prepared,rgb,gold)
    prepared,rgb,gold=fault_fixture;model=engine.model;faults=[]
    forward=model.sampler.pre_guide.forward
    with patch.object(model.sampler.pre_guide,'forward',side_effect=lambda c:forward(c)*float('nan')):
        try:prepared.apply(rgb,1)
        except ValueError:faults.append('nonfinite guide rejected before image return')
        else:raise ValueError('Invalid guide escaped')
    apply=model.apply_coeffs
    with patch.object(model,'apply_coeffs',side_effect=lambda coeff,c:apply(coeff,c)*float('nan')):
        try:prepared.apply(rgb,1)
        except ValueError:faults.append('nonfinite raw rejected before image return')
        else:raise ValueError('Invalid raw escaped')
    calls=[0]
    def late(coeff,c):
        calls[0]+=1
        value=apply(coeff,c)
        return value*float('nan') if calls[0]==2 else value
    with patch.object(model,'apply_coeffs',side_effect=late):
        try:prepared.apply(rgb,1)
        except ValueError:faults.append('nonfinite late band rejected before image return')
        else:raise ValueError('Invalid later band escaped')
    if not np.array_equal(prepared.apply(rgb,1),gold):raise ValueError('CUDA unusable or output changed after controlled faults')
    report.update(faults=faults,cuda_recovered_pixel_exact=True,passed=True)
    if any(digest(Path(__file__).with_name(n))!=v for n,v in report['source_sha256'].items()):raise ValueError('Source changed')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,profile=args.profile,cases=6,strength_cases=18,faults=3)),flush=True)


if __name__=='__main__':main()
