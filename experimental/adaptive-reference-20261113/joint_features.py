"""Development-only luma-conditioned chroma features; no semantic masks."""
import math
import numbers
import torch
from training_checks import finite_output


def conditional_color(ycbcr,num_bins=128,prior_count=32.):
    if (not isinstance(ycbcr,torch.Tensor) or ycbcr.ndim!=4 or ycbcr.shape[1]!=3
            or min(ycbcr.shape)<1 or ycbcr.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(ycbcr).all()):raise ValueError('Expected finite nonempty floating YCbCr BCHW')
    if type(num_bins)!=int or num_bins<1 or isinstance(prior_count,bool) or not isinstance(prior_count,numbers.Real) or not math.isfinite(prior_count) or prior_count<0:
        raise ValueError('Invalid conditional statistics configuration')
    if (ycbcr[:,0]<0).any() or (ycbcr[:,0]>1).any() or (ycbcr[:,1:].abs()>.5).any():raise ValueError('YCbCr outside declared range')
    batch=ycbcr.shape[0];flat=ycbcr.flatten(2);indices=(flat[:,:1]*num_bins).floor().clamp(0,num_bins-1).long()
    count=torch.zeros((batch,1,num_bins),device=ycbcr.device,dtype=ycbcr.dtype).scatter_add(2,indices,torch.ones_like(flat[:,:1]))
    chroma=flat[:,1:];moments=torch.cat([2*chroma,math.sqrt(2)*chroma.square().sum(1,keepdim=True).sqrt()],1)
    sums=torch.zeros((batch,3,num_bins),device=ycbcr.device,dtype=ycbcr.dtype).scatter_add(2,indices.expand(-1,3,-1),moments)
    return finite_output(sums/(count+float(prior_count)).clamp_min(1e-12),(batch,3,num_bins))


def enable_joint_input(model):
    old=model.hist_projection[0]
    if not isinstance(old,torch.nn.Conv1d) or old.in_channels!=6 or old.groups!=1:raise ValueError('Expected unmodified six-channel author projection')
    new=torch.nn.Conv1d(12,old.out_channels,old.kernel_size,stride=old.stride,padding=old.padding,dilation=old.dilation,
        bias=old.bias is not None,padding_mode=old.padding_mode,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_();new.weight[:,:6].copy_(old.weight)
        if old.bias is not None:new.bias.copy_(old.bias)
    new.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:new.bias.requires_grad_(old.bias.requires_grad)
    model.hist_projection[0]=new


def joint_forward(model,api,content,reference,disable_joint=False):
    if content.shape[0]!=reference.shape[0]:raise ValueError('Batch sizes differ')
    c,s=api['srgb_to_ycbcr'](content),api['srgb_to_ycbcr'](reference)
    jc,js=conditional_color(c),conditional_color(s)
    hc,hs=api['batch_histogram_1D'](c),api['batch_histogram_1D'](s)
    if type(disable_joint)!=bool:raise ValueError('Invalid ablation flag')
    if disable_joint:jc,js=torch.zeros_like(jc),torch.zeros_like(js)
    features=torch.cat([hc,hs,jc,js],1)
    low=api['TF'].resize(content,(64,64),antialias=True)
    grid=finite_output(api['lowpass'](model.apply_grid(low,features)),(content.shape[0],13,8,16,16))
    raw=finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape)
    return raw,grid
