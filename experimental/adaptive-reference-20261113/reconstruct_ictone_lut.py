"""Restore pinned author's default LUT reconstruction without model reruns/GT."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

from ictone_flux_stage import ROOT
from screen_ictone_first import digest


def author_functions():
    import numpy as np
    import torch
    source=ROOT/'.tools/ICTone-audit/inference.py'
    assert digest(source)=='0899f20276cfe6e3da42426fc511a96644440b49c721824d188e37760ee1bcca'
    names={'estimate_lut','_estimate_lut_gpu','apply_lut','_apply_lut_gpu','_apply_lut_cpu'}
    tree=ast.parse(source.read_bytes())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(functions)==5
    namespace={'np':np,'torch':torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--screen',required=True)
    p.add_argument('--run',required=True)
    args=p.parse_args()
    assert all(Path(x).name==x and x not in ('.','..') for x in [args.screen,args.run])
    import numpy as np
    import torch
    from PIL import Image
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    screen=base/args.screen
    prior=json.loads((screen/'results.json').read_text())
    assert prior['finished'], 'wait for original GPU generation to finish'
    assert json.loads((screen/'audit.json').read_text())['finished']
    out=base/args.run
    out.mkdir(exist_ok=False)
    funcs=author_functions()
    # Portable application check only: identical LUT must preserve RGB order.
    g=np.linspace(0,255,33,dtype=np.float32)
    identity=np.stack(np.meshgrid(g,g,g,indexing='ij'),axis=-1)
    before=np.random.default_rng(0).integers(0,256,(16,16,3),dtype=np.uint8)
    cpu=funcs['apply_lut'](before,identity,device='cpu')
    gpu=funcs['apply_lut'](before,identity,device='cuda')
    assert np.abs(cpu.astype(int)-before).max()<=1 and np.abs(gpu.astype(int)-before).max()<=1
    assert np.abs(cpu.astype(int)-gpu).max()<=1
    result=dict(finished=False,scope='author default reconstruction on completed consumed subset only; no GT used',
                screen=args.screen,screen_result_sha256=digest(screen/'results.json'),
                source_sha256=digest(Path(__file__)),author_sha256=digest(ROOT/'.tools/ICTone-audit/inference.py'),
                protocol_sha256=digest(ROOT/'docs/research/2026-11-17-ictone-lut.md'),
                config=dict(size=33,lam_smooth=.05,lam_anchor=1e-5,tol=1e-4,max_iter=200,backend='cuda'),
                application_checks=dict(identity_max_error=int(np.abs(gpu.astype(int)-before).max()),
                                        cpu_gpu_max_error=int(np.abs(cpu.astype(int)-gpu).max())),rows=[])
    (out/'protocol-frozen.md').write_bytes((ROOT/'docs/research/2026-11-17-ictone-lut.md').read_bytes())
    (out/'reconstruct_ictone_lut.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for item in prior['completed']:
        assert time.monotonic()-started<120,'120-second reconstruction budget exhausted'
        folder=base/item['run']
        assert digest(folder/'metric256.png')==item['metric_sha256']
        with Image.open(folder/'content256.png') as im:content=np.array(im.convert('RGB'))
        with Image.open(folder/'native512.png') as im:after=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        then=time.monotonic()
        lut=funcs['estimate_lut'](content.reshape(-1,3),after.reshape(-1,3),size=33,device='cuda')
        assert lut.shape==(33,33,33,3) and np.isfinite(lut).all()
        output=funcs['apply_lut'](content,lut,device='cuda')
        torch.cuda.synchronize()
        assert output.shape==content.shape and output.dtype==np.uint8
        index=item['index']
        np.save(out/(str(index)+'-lut.npy'),lut)
        path=out/(str(index)+'.png')
        Image.fromarray(output).save(path)
        result['rows'].append(dict(index=index,source_run=item['run'],content_sha256=digest(folder/'content256.png'),
                                   native_sha256=digest(folder/'native512.png'),output_sha256=digest(path),
                                   lut_sha256=digest(out/(str(index)+'-lut.npy')),seconds=time.monotonic()-then))
        (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result['rows'][-1]),flush=True)
    elapsed=time.monotonic()-started
    assert elapsed<=120,elapsed
    result.update(finished=True,wall_reconstruction_seconds=elapsed,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
