"""Fixed-radius appearance context; no new model parameters or GT inference."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from appearance_pairwise import appearance_forward
from pairwise_context import pairwise_histograms
from training_checks import finite_output


def valid_radius(radius):
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=1e-8:raise ValueError('Expected fixed positive finite radius')


def normalized_map(features,radius):
    valid_radius(radius)
    if not isinstance(features,torch.Tensor) or features.ndim!=4 or features.shape[1]!=128 or min(features.shape)<1 or features.dtype not in (torch.float32,torch.float64) or not torch.isfinite(features).all():raise ValueError('Expected finite B128HW float32/64 feature map')
    mean=features.mean((-2,-1),keepdim=True)
    return finite_output(F.normalize(mean,p=2,dim=1,eps=1e-8)*radius,(features.shape[0],128,1,1))


def calibrated_radius(norms):
    values=np.asarray(norms,dtype=np.float64)
    if values.ndim!=1 or values.size<1 or not np.isfinite(values).all() or (values<0).any():raise ValueError('Expected finite nonnegative input norms')
    value=float(np.median(values));valid_radius(value);return value


def radius_forward(model,api,content,reference,radius,normalized=True):
    valid_radius(radius)
    if type(normalized)!=bool:raise ValueError('Expected boolean normalization mode')
    if not normalized:return appearance_forward(model,api,content,reference)
    p=next(model.parameters())
    for value in [content,reference]:
        if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=3 or min(value.shape)<1 or value.shape[0]!=content.shape[0] or value.dtype!=p.dtype or value.device!=p.device or not torch.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Expected finite unit RGB batches on model device')
    c,s=api['srgb_to_ycbcr'](content),api['srgb_to_ycbcr'](reference);hc,hs=api['batch_histogram_1D'](c),api['batch_histogram_1D'](s);source=model.img_projection(api['TF'].resize(content,(64,64),antialias=True));ref_features=model.img_projection(api['TF'].resize(reference,(64,64),antialias=True))
    context=model.hist_projection(torch.cat([hc,hs],1))+model.pairwise_adapter(pairwise_histograms(c),pairwise_histograms(s))[:,:,None]
    context=context+model.reference_adapter(normalized_map(source,radius),normalized_map(ref_features,radius))[:,:,None]
    raw=model.fusion(torch.cat([source,model.attention(source,context)],1)).reshape(-1,13,8,16,16);alpha=F.softplus(raw[:,-1:])+1e-3;grid=finite_output(api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1)),(content.shape[0],13,8,16,16))
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),grid
