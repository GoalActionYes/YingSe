"""Research-only reference appearance context using the existing image encoder."""
import torch
import torch.nn.functional as F
from training_checks import finite_output


class ReferenceContext(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(384,128),torch.nn.GELU(),torch.nn.Linear(128,128))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)
    def forward(self,source,reference):
        for value in [source,reference]:
            if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=128 or min(value.shape)<1 or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('Expected finite nonempty B128HW image features')
        if source.shape[0]!=reference.shape[0]:raise ValueError('Feature batches differ')
        a,b=source.mean((-2,-1)),reference.mean((-2,-1))
        return finite_output(self.layers(torch.cat([a,b,b-a],1)),(a.shape[0],128))


def enable_reference_context(model):
    if hasattr(model,'reference_adapter'):raise ValueError('Reference context already installed')
    parameter=next(model.parameters())
    model.reference_adapter=ReferenceContext().to(device=parameter.device,dtype=parameter.dtype)


def reference_grid(model,api,content,reference,disable_reference=False):
    for value in [content,reference]:
        if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=3 or min(value.shape)<1 or not torch.isfinite(value).all():raise ValueError('Expected finite RGB BCHW')
    if content.shape[0]!=reference.shape[0] or type(disable_reference)!=bool:raise ValueError('Invalid reference context inputs')
    hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](content));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](reference))
    low=api['TF'].resize(content,(64,64),antialias=True);source_features=model.img_projection(low)
    context=model.hist_projection(torch.cat([hc,hs],1))
    if not disable_reference:
        ref_low=api['TF'].resize(reference,(64,64),antialias=True)
        ref_features=model.img_projection(ref_low)
        context=context+model.reference_adapter(source_features,ref_features)[:,:,None]
    fused=torch.cat([source_features,model.attention(source_features,context)],1)
    raw_grid=model.fusion(fused).reshape(-1,13,8,16,16)
    alpha=F.softplus(raw_grid[:,-1:])+1e-3
    grid=api['lowpass'](torch.cat([raw_grid[:,:-1]*alpha,alpha],1))
    return finite_output(grid,(content.shape[0],13,8,16,16))


def reference_forward(model,api,content,reference,disable_reference=False):
    grid=reference_grid(model,api,content,reference,disable_reference)
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),grid
