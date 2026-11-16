"""Predict local correction with explicitly observed baseline color, research only."""
import torch
import torch.nn.functional as F
from semantic_local import match


def color_context(rgb,baseline,control):
    if type(control)!=bool or rgb.ndim!=4 or rgb.shape[1]!=3 or baseline.shape!=rgb.shape or not torch.isfinite(rgb).all() or not torch.isfinite(baseline).all():raise ValueError('Invalid baseline context')
    value=rgb if control else baseline.clamp(0,1)
    return F.adaptive_avg_pool2d(value,(8,8)).flatten(2).transpose(1,2)


class BaselineContextCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(777,128),torch.nn.GELU(),torch.nn.Linear(128,12))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self,rgb,baseline,content,reference,control=False):
        colors=color_context(rgb,baseline,control);context=match(content,reference)
        field=self.layers(torch.cat([content,context,colors],-1)).transpose(1,2).reshape(-1,12,8,8)
        field=F.interpolate(field,size=rgb.shape[2:],mode='bilinear',align_corners=False).reshape(-1,3,4,*rgb.shape[2:])
        x=torch.cat([rgb,torch.ones_like(rgb[:,:1])],1)
        result=baseline+.1*torch.tanh((field*x[:,None]).sum(2))
        if not torch.isfinite(result).all():raise ValueError('Nonfinite baseline-aware output')
        return result
