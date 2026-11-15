"""Frozen PETAL z extraction for an offline diagnostic, never a product engine."""
import torch
import torch.nn.functional as F
from petal_model import tensor


@torch.inference_mode()
def encode(model, images):
    if not images:
        raise ValueError('At least one RGB8 input required')
    lab=torch.cat([model.api.rgb2lab(F.interpolate(tensor(v,model.device),size=(512,512),mode='bilinear',align_corners=False)) for v in images])
    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
        z=model.style(lab)['z']
    if z.ndim!=2 or z.shape[0]!=len(images) or z.shape[1]<1 or not torch.isfinite(z).all() or (torch.linalg.vector_norm(z,dim=1)<=1e-12).any():
        raise ValueError('Invalid or zero photographic style descriptor')
    return F.normalize(z,dim=-1)
