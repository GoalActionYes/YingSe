"""Training-only latent-field loss, no extra inference dependency."""
import torch
import torch.nn.functional as F
from regional_correction import finite


def latent_field(head,rgb,probability,content,reference):
    coefficients=head.layers(torch.cat([content,reference,reference-content],1)).reshape(-1,3,3,4)
    x=torch.cat([rgb,torch.ones_like(rgb[:,:1])],1)
    parts=torch.einsum('bkij,bjhw->bkihw',coefficients,x)
    value=parts[:,0]+probability*parts[:,1]+(1-probability)*parts[:,2]
    finite(value,rgb.shape);return value


def field_loss(prediction,target):
    finite(prediction,target.shape);finite(target,prediction.shape)
    return F.smooth_l1_loss(prediction,target.detach(),beta=1.)
