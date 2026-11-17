"""Offline first-decomposition audit, independent NNLS and sRGB equations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/Marigold-IID-audit'
OUT=ROOT/'outputs/adaptive-reference-20261113/intrinsic-prior-first-v1'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def main():
    sys.path.insert(0,str(ROOT/'.tools/DeepWB-audit/vendor'))
    from scipy.optimize import nnls
    d=json.loads((OUT/'results.json').read_text());assert d['finished'] and d['exact_repeat'] and not d['product_integrated']
    for n,h in d['source_sha256'].items():assert digest(OUT/n)==h
    for entry in d['assets']['files']:assert digest(ASSETS/'models'/entry['path'])==entry['sha256']
    parent=ROOT/'outputs/adaptive-reference-20261113/independent-coco/9483-input.png'
    assert digest(parent)==d['parent_sha256'] and digest(OUT/'input.png')==d['input_sha256']
    rgb=np.array(Image.open(parent).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    assert np.array_equal(rgb,np.array(Image.open(OUT/'input.png')))
    assert d['config']==dict(steps=4,ensemble_size=1,processing_resolution=768,batch_size=1,seed=20261117)
    results=[]
    for run in d['runs']:
        p=OUT/run['file'];assert digest(p)==run['sha256'];a=np.load(p,allow_pickle=False)
        assert a.shape==(3,256,256,3) and np.isfinite(a).all()
        assert all(float(v.min())==s['min'] and float(v.max())==s['max'] for v,s in zip(a,run['ranges'],strict=True))
        results.append(a)
    assert len(results)==2 and np.array_equal(*results)
    x=rgb.astype(np.float64)/255;linear=np.empty_like(x);low=x<=.04045
    linear[low]=x[low]/12.92;linear[~low]=((x[~low]+.055)/1.055)**2.4
    albedo,shading,residual=results[0].astype(np.float64)
    design=np.column_stack(((albedo*shading).ravel(),residual.ravel()))
    scales,_=nnls(design,linear.ravel());raw=(design@scales).reshape(rgb.shape)
    clipped=np.clip(raw,0,1);srgb=np.empty_like(clipped);low=clipped<=.0031308
    srgb[low]=clipped[low]*12.92;srgb[~low]=1.055*clipped[~low]**(1/2.4)-.055
    png=np.rint(srgb*255).astype(np.uint8)
    summary=dict(scales=scales.tolist(),linear_mae=float(np.mean(np.abs(raw-linear))),
                 linear_rmse=float(np.sqrt(np.mean((raw-linear)**2))),raw_outside_fraction=float(np.mean((raw<0)|(raw>1))),
                 srgb_mae=float(np.mean(np.abs(png.astype(float)-rgb))))
    assert all(np.allclose(v,d['reconstruction'][k],atol=1e-9,rtol=0) for k,v in summary.items())
    assert digest(OUT/'reconstruction.png')==d['reconstruction_sha256']
    assert np.array_equal(png,np.array(Image.open(OUT/'reconstruction.png')))
    audit=dict(finished=True,raw_repeat_exact=True,independent_scipy_nnls=True,reconstruction=summary,
               results_sha256=digest(OUT/'results.json'),scope='file identity and offline reconstruction; not independent GPU or intrinsic ground truth')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))

if __name__=='__main__':main()
