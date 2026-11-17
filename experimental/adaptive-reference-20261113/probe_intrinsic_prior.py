"""Pinned first intrinsic decomposition, no reference transfer or product change."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/Marigold-IID-audit'
OUT=ROOT/'outputs/adaptive-reference-20261113/intrinsic-prior-first-v1'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def to_linear(a):return np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)

def to_srgb(a):
    x=a.clip(0,1)
    return np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055)

def reconstruct(prediction,rgb):
    a,s,r=prediction.astype(np.float64)
    x=np.stack(((a*s).ravel(),r.ravel()),1);y=to_linear(rgb.astype(float)/255).ravel()
    choices=[np.zeros(2)]
    both=np.linalg.lstsq(x,y,rcond=None)[0]
    if (both>=0).all():choices.append(both)
    for j in range(2):
        c=np.zeros(2);den=x[:,j]@x[:,j]
        if den:c[j]=max(0.,float(x[:,j]@y/den))
        choices.append(c)
    c=min(choices,key=lambda b:float(np.square(x@b-y).mean()))
    raw=(x@c).reshape(rgb.shape)
    png=np.rint(to_srgb(raw)*255).astype(np.uint8)
    return png,dict(scales=c.tolist(),linear_mae=float(np.abs(raw-to_linear(rgb/255)).mean()),
                    linear_rmse=float(np.sqrt(np.square(raw-to_linear(rgb/255)).mean())),
                    raw_outside_fraction=float(((raw<0)|(raw>1)).mean()),
                    srgb_mae=float(np.abs(png.astype(float)-rgb).mean()))

def main():
    OUT.mkdir(exist_ok=False)
    for p in [ROOT/'.tools/DeepWB-audit/vendor',ROOT/'.tools/StyleGallery-audit/vendor']:
        sys.path.insert(0,str(p))
    import torch,diffusers,transformers
    from diffusers import MarigoldIntrinsicsPipeline
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((ASSETS/'assets.json').read_text());assert manifest['finished']
    for f in manifest['files']:assert digest(ASSETS/'models'/f['path'])==f['sha256']
    parent=ROOT/'outputs/adaptive-reference-20261113/independent-coco/9483-input.png'
    rgb=np.array(Image.open(parent).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    Image.fromarray(rgb).save(OUT/'input.png')
    names=['probe_intrinsic_prior.py','intrinsic-prior-protocol.md']
    sources={n:digest(Path(__file__).with_name(n)) for n in names}
    for n in names:(OUT/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    implementation=Path(diffusers.__file__).parent/'pipelines/marigold'
    report=dict(scope='first offline intrinsic decomposition, not color transfer or product quality',assets=manifest,
        parent_sha256=digest(parent),input_sha256=digest(OUT/'input.png'),source_sha256=sources,
        implementation_sha256={p.name:digest(p) for p in implementation.glob('*.py')},
        versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,numpy=np.__version__),
        config=dict(steps=4,ensemble_size=1,processing_resolution=768,batch_size=1,seed=20261117),
        runs=[],finished=False,product_integrated=False)
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    start=time.perf_counter()
    pipe=MarigoldIntrinsicsPipeline.from_pretrained(ASSETS/'models',variant='fp16',torch_dtype=torch.float16,
        use_safetensors=True,low_cpu_mem_usage=True,local_files_only=True).to('cuda')
    torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-start
    assert list(pipe.config.target_properties['target_names'])==['albedo','shading','residual']
    pipe.set_progress_bar_config(disable=True)
    first=None
    for i in range(2):
        assert time.perf_counter()-start<1200
        generator=torch.Generator(device='cuda').manual_seed(20261117);t=time.perf_counter()
        with torch.inference_mode():result=pipe(Image.fromarray(rgb),num_inference_steps=4,ensemble_size=1,
            processing_resolution=768,batch_size=1,generator=generator,output_type='np').prediction
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        assert seconds<=600 and result.shape==(3,256,256,3) and np.isfinite(result).all()
        p=OUT/f'prediction-{i}.npy';np.save(p,result,allow_pickle=False)
        if first is None:first=result.copy()
        else:assert np.array_equal(first,result),'Repeated raw prediction differs'
        report['runs'].append(dict(file=p.name,sha256=digest(p),seconds=seconds,
                                   ranges=[dict(min=float(v.min()),max=float(v.max())) for v in result]))
        (OUT/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report['runs'][-1]),flush=True)
    reconstruction,stats=reconstruct(first,rgb);report['reconstruction']=stats
    Image.fromarray(reconstruction).save(OUT/'reconstruction.png')
    panel=Image.new('RGB',(1280,282),'white');draw=ImageDraw.Draw(panel)
    tiles=[rgb]+[np.rint(to_srgb(v)*255).astype(np.uint8) for v in first]+[reconstruction]
    for i,(label,a) in enumerate(zip(['input','albedo','shading (unscaled)','residual (unscaled)','two-scale reconstruction'],tiles,strict=True)):
        draw.text((i*256+4,4),label,fill='black');panel.paste(Image.fromarray(a),(i*256,26))
    panel.save(OUT/'diagnostic-panel.png')
    report.update(finished=True,exact_repeat=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  reconstruction_sha256=digest(OUT/'reconstruction.png'),seconds=time.perf_counter()-start)
    assert all(digest(Path(__file__).with_name(n))==h for n,h in sources.items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(finished=True,exact_repeat=True,reconstruction=stats,seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
