"""Unintegrated inference diagnostic; region histograms are out of training domain."""
import numpy as np
from PIL import Image
import torch
from spatial_render import PreparedSpatial,validate_rgb
from training_checks import finite_output


def probability(value,shape=None):
    p=np.asarray(value,dtype=np.float32)
    if p.ndim!=2 or not p.size or (shape is not None and p.shape!=shape) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Expected finite matching probability map in 0..1')
    return p


def supported(target,reference):
    for value in [target,reference]:
        p=probability(value)
        if min(float(p.mean()),float((1-p).mean()))<.02 or not .02<=float((p>.5).mean())<=.98:
            return False
    return True


def weighted_histogram(ycbcr,weights):
    if ycbcr.ndim!=4 or ycbcr.shape[1]!=3 or weights.shape!=(ycbcr.shape[0],1,*ycbcr.shape[2:]):
        raise ValueError('Expected BCHW YCbCr and B1HW weights')
    if not torch.isfinite(ycbcr).all() or not torch.isfinite(weights).all() or torch.any(weights<0) or torch.any(weights>1) or torch.any(weights.sum((2,3))<=0):
        raise ValueError('Invalid histogram values or empty support')
    shifted=ycbcr.clone();shifted[:,1:]+=.5
    coords=(shifted.flatten(2)*128).floor().clamp(0,127).long()
    hist=torch.zeros(ycbcr.shape[0],3,128,dtype=ycbcr.dtype,device=ycbcr.device)
    hist.scatter_add_(2,coords,weights.flatten(2).expand(-1,3,-1))
    return hist/hist.sum(2,keepdim=True)


def blend(foreground,background,mask):
    if foreground.shape!=background.shape or foreground.ndim!=3 or foreground.shape[2]!=3 or not np.isfinite(foreground).all() or not np.isfinite(background).all():
        raise ValueError('Expected finite matching RGB float outputs')
    p=probability(mask,foreground.shape[:2])[...,None]
    return np.rint(np.clip(foreground*p+background*(1-p),0,1)*255).astype(np.uint8)


@torch.inference_mode()
def predict(engine,target,reference,target_mask,reference_mask,baseline):
    validate_rgb(target);validate_rgb(reference);validate_rgb(baseline)
    p=probability(target_mask,target.shape[:2]);r=probability(reference_mask,reference.shape[:2])
    if baseline.shape!=target.shape:raise ValueError('Baseline dimensions changed')
    if not supported(p,r):return baseline.copy(),dict(applied=False,reason='Insufficient reliable two-region support')
    c,s=engine.analysis(target),engine.analysis(reference)
    yc,ys=[engine.api['srgb_to_ycbcr'](v) for v in (c,s)]
    masks=[torch.from_numpy(np.array(Image.fromarray(v).resize((256,256),Image.Resampling.BILINEAR)))[None,None].to(engine.device) for v in (p,r)]
    low=engine.api['TF'].resize(c,(64,64),antialias=True)
    outputs=[]
    for foreground in [True,False]:
        wc,ws=[m if foreground else 1-m for m in masks]
        hc,hs=weighted_histogram(yc,wc),weighted_histogram(ys,ws)
        grid=engine.api['lowpass'](engine.model.apply_grid(low,torch.cat([hc,hs],1)))
        finite_output(grid,(1,13,8,16,16))
        outputs.append(PreparedSpatial(engine,grid,{}).float_output(target))
    return blend(*outputs,p),dict(applied=True,target_person_mass=float(p.mean()),reference_person_mass=float(r.mean()))
