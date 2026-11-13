"""Reference appearance plus joint color distributions; research-only conditioning."""
import torch
import torch.nn.functional as F
from pairwise_context import pairwise_grid,pairwise_histograms
from reference_context import enable_reference_context
from training_checks import finite_output


def enable_appearance_context(model):
    if not hasattr(model,'pairwise_adapter'):raise ValueError('Install pairwise context first')
    enable_reference_context(model)


def appearance_grid(model,api,content,reference,disable_appearance=False,appearance_reference=None):
    if type(disable_appearance)!=bool or (disable_appearance and appearance_reference is not None):raise ValueError('Invalid appearance mode')
    for value in [content,reference,*([appearance_reference] if appearance_reference is not None else [])]:
        if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=3 or min(value.shape)<1 or not value.is_floating_point() or not torch.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Expected finite normalized RGB BCHW')
        if value.shape[0]!=content.shape[0]:raise ValueError('Input batches differ')
    if disable_appearance:return pairwise_grid(model,api,content,reference)
    c,s=api['srgb_to_ycbcr'](content),api['srgb_to_ycbcr'](reference)
    hc,hs=api['batch_histogram_1D'](c),api['batch_histogram_1D'](s)
    source=model.img_projection(api['TF'].resize(content,(64,64),antialias=True))
    context=model.hist_projection(torch.cat([hc,hs],1))
    context=context+model.pairwise_adapter(pairwise_histograms(c),pairwise_histograms(s))[:,:,None]
    image=reference if appearance_reference is None else appearance_reference
    features=model.img_projection(api['TF'].resize(image,(64,64),antialias=True))
    context=context+model.reference_adapter(source,features)[:,:,None]
    fused=torch.cat([source,model.attention(source,context)],1);raw=model.fusion(fused).reshape(-1,13,8,16,16)
    alpha=F.softplus(raw[:,-1:])+1e-3
    return finite_output(api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1)),(content.shape[0],13,8,16,16))


def appearance_forward(model,api,content,reference,disable_appearance=False,appearance_reference=None):
    grid=appearance_grid(model,api,content,reference,disable_appearance,appearance_reference)
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),grid
