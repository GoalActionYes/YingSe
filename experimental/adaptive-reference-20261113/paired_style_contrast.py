"""Real-pair training supervision; never accepts GT in inference."""
import math
import torch
import torch.nn.functional as F


def paired_style_contrast(reference,truth,temperature=.1):
    for value in [reference,truth]:
        if not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[0]!=4 or value.shape[1]<1 or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('Expected finite floating four-row style vectors')
    if reference.shape!=truth.shape or reference.dtype!=truth.dtype or reference.device!=truth.device or type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid paired style contrast inputs')
    a=F.normalize(reference,p=2,dim=1,eps=1e-8).reshape(2,2,-1);b=F.normalize(truth,p=2,dim=1,eps=1e-8).reshape(2,2,-1)
    logits=a@b.transpose(1,2)/temperature;labels=torch.arange(2,device=reference.device).repeat(2)
    return (F.cross_entropy(logits.reshape(4,2),labels)+F.cross_entropy(logits.transpose(1,2).reshape(4,2),labels))/2
