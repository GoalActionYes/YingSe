"""Three pairwise Y/Cb/Cr histograms, not semantic reference correspondence."""
import torch
import torch.nn.functional as F
from training_checks import finite_output


def pairwise_histograms(ycbcr,bins=16):
    if type(bins)!=int or not 2<=bins<=32:raise ValueError('Expected 2..32 bins')
    if not isinstance(ycbcr,torch.Tensor) or ycbcr.ndim!=4 or ycbcr.shape[1]!=3 or min(ycbcr.shape)<1 or ycbcr.dtype not in (torch.float32,torch.float64) or not torch.isfinite(ycbcr).all():raise ValueError('Expected finite nonempty floating YCbCr BCHW')
    if (ycbcr[:,0]<-1e-6).any() or (ycbcr[:,0]>1+1e-6).any() or (ycbcr[:,1:].abs()>.5+1e-6).any():raise ValueError('YCbCr outside range')
    flat=ycbcr.flatten(2);normalized=torch.cat([flat[:,:1],flat[:,1:]+.5],1).clamp(0,1)
    indices=(normalized*bins).floor().clamp(0,bins-1).long();parts=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        locations=indices[:,a]*bins+indices[:,b]
        counts=torch.zeros((flat.shape[0],bins*bins),device=flat.device,dtype=flat.dtype).scatter_add(1,locations,torch.ones_like(locations,dtype=flat.dtype))
        parts.append((counts/flat.shape[-1]).sqrt())
    return finite_output(torch.cat(parts,1),(flat.shape[0],3*bins*bins))


class PairwiseContext(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(2304,128),torch.nn.GELU(),torch.nn.Linear(128,128))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self,source,reference):
        for value in [source,reference]:
            if not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[1]!=768 or value.shape[0]<1 or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('Expected finite B768 features')
        if source.shape!=reference.shape:raise ValueError('Feature batch dimensions differ')
        return finite_output(self.layers(torch.cat([source,reference,reference-source],1)),(source.shape[0],128))


def enable_pairwise_context(model):
    if hasattr(model,'pairwise_adapter'):raise ValueError('Pairwise context already enabled')
    parameter=next(model.parameters());model.pairwise_adapter=PairwiseContext().to(device=parameter.device,dtype=parameter.dtype)


def pairwise_grid(model,api,content,reference,disable_pairwise=False):
    if content.shape[0]!=reference.shape[0] or type(disable_pairwise)!=bool:raise ValueError('Invalid paired inputs')
    c,s=api['srgb_to_ycbcr'](content),api['srgb_to_ycbcr'](reference)
    hc,hs=api['batch_histogram_1D'](c),api['batch_histogram_1D'](s)
    low=api['TF'].resize(content,(64,64),antialias=True);source=model.img_projection(low)
    context=model.hist_projection(torch.cat([hc,hs],1))
    if not disable_pairwise:context=context+model.pairwise_adapter(pairwise_histograms(c),pairwise_histograms(s))[:,:,None]
    fused=torch.cat([source,model.attention(source,context)],1);raw=model.fusion(fused).reshape(-1,13,8,16,16)
    alpha=F.softplus(raw[:,-1:])+1e-3;grid=api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1))
    return finite_output(grid,(content.shape[0],13,8,16,16))


def pairwise_forward(model,api,content,reference,disable_pairwise=False):
    grid=pairwise_grid(model,api,content,reference,disable_pairwise)
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),grid
