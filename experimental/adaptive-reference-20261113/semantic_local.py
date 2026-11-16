"""Frozen feature correspondence plus learned local color field, research only."""
import torch
import torch.nn.functional as F


def validate(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(64,387) or not x.is_floating_point() or not torch.isfinite(x).all():raise ValueError('Expected finite B64x387 tokens')


def match(content,reference,pooled=False):
    validate(content);validate(reference)
    if content.shape!=reference.shape or type(pooled)!=bool:raise ValueError('Token batches differ')
    if pooled:reference=reference.mean(1,keepdim=True).expand_as(reference)
    q=F.normalize(content[...,:384],dim=-1);k=F.normalize(reference[...,:384],dim=-1)
    weights=torch.softmax(torch.bmm(q,k.transpose(1,2))*10,dim=-1)
    return torch.bmm(weights,reference)


class SemanticCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(774,128),torch.nn.GELU(),torch.nn.Linear(128,12))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self,rgb,baseline,content,reference,pooled=False):
        if rgb.ndim!=4 or rgb.shape[1]!=3 or baseline.shape!=rgb.shape or not torch.isfinite(rgb).all() or not torch.isfinite(baseline).all():raise ValueError('Invalid input image')
        context=match(content,reference,pooled)
        field=self.layers(torch.cat([content,context],-1)).transpose(1,2).reshape(-1,12,8,8)
        field=F.interpolate(field,size=rgb.shape[2:],mode='bilinear',align_corners=False).reshape(-1,3,4,*rgb.shape[2:])
        x=torch.cat([rgb,torch.ones_like(rgb[:,:1])],1)
        residual=(field*x[:,None]).sum(2)
        result=baseline+.1*torch.tanh(residual)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite semantic correction')
        return result
