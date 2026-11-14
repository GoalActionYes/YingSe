"""Frame-resident full-strength RGB8 rendering; invalid frames never escape."""
import numpy as np
import torch
import torch.nn.functional as F
from spatial_render import coordinates,validate_rgb


def finite_or_zero(value,shape):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shape):
        raise ValueError('Network output shape validation failed before clipping')
    finite=torch.isfinite(value)
    return torch.where(finite,value,0.),finite.all()


@torch.inference_mode()
def render(prepared,rgb,chunk_rows=256):
    validate_rgb(rgb)
    if type(chunk_rows)!=int or chunk_rows<1:raise ValueError('Positive integer band height required')
    model=prepared.engine.model;device=prepared.grid.device;height,width=rgb.shape[:2]
    source=torch.from_numpy(rgb.copy()).to(device=device)
    output=torch.empty(source.shape,device=device,dtype=torch.uint8)
    flags=[]
    weights=F.softmax(model.sampler.L_weights,dim=1)
    for start in range(0,height,chunk_rows):
        stop=min(start+chunk_rows,height)
        c=source[start:stop].permute(2,0,1)[None].to(dtype=torch.float32)/255
        z=(model.sampler.pre_guide(c)*weights).sum(1)
        z,z_valid=finite_or_zero(z,(1,stop-start,width));flags.append(z_valid)
        coords=coordinates(width,height,start,stop,z)
        coeff=F.grid_sample(prepared.grid,coords,mode='bilinear',align_corners=False,padding_mode='border').squeeze(2)
        raw,raw_valid=finite_or_zero(model.apply_coeffs(coeff,c),c.shape);flags.append(raw_valid)
        # Only finite values are ever quantized, even on a frame later rejected.
        output[start:stop]=raw[0].clamp(0,1).mul(255).round().to(torch.uint8).permute(1,2,0)
    if not torch.stack(flags).all().item():raise ValueError('Network output nonfinite validation failed before image return')
    return output.cpu().numpy()
