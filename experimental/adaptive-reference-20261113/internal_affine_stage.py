"""Observe the pinned author affine stage without changing its forward path."""
import torch


def internal_affine(model,coeffs,content):
    if not isinstance(content,torch.Tensor) or content.ndim!=4 or content.shape[1]!=3 or min(content.shape)<1 or not content.is_floating_point() or not torch.isfinite(content).all():raise ValueError('Expected finite RGB BCHW')
    b,_,h,w=content.shape
    if not isinstance(coeffs,torch.Tensor) or tuple(coeffs.shape)!=(b,13,h,w) or not coeffs.is_floating_point() or coeffs.dtype!=content.dtype or coeffs.device!=content.device or not torch.isfinite(coeffs).all() or (coeffs[:,-1:]<=0).any():raise ValueError('Expected finite positive-alpha13coefficients')
    values=(coeffs[:,:-1]/coeffs[:,-1:]).flatten(2).permute(0,2,1)
    mats=values[...,:9].reshape(b*h*w,3,3);bias=values[...,9:].reshape(b*h*w,3,1)
    image=model.pre_guide((content-.5)*2.).flatten(2).permute(0,2,1)[...,None].reshape(b*h*w,3,1)
    return (torch.bmm(mats,image)+bias).reshape(b,h,w,3).permute(0,3,1,2)
