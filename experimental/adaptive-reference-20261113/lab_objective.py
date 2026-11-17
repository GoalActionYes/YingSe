"""Differentiable sRGB -> D50-relative CIELAB; main-image objective probe."""
import torch
import torch.nn.functional as F


def lab(rgb):
    if not isinstance(rgb,torch.Tensor) or rgb.ndim!=4 or rgb.shape[1]!=3 or min(rgb.shape)<1 or rgb.dtype not in [torch.float32,torch.float64] or not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():raise ValueError('Expected finite unit BCHW sRGB')
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb.clamp_min(.04045)+.055)/1.055).pow(2.4))
    matrix=rgb.new_tensor([[506752/1228815,87881/245763,12673/70218],[87098/409605,175762/245763,12673/175545],[7918/409605,87881/737289,1001167/1053270]])
    xyz=torch.einsum('ij,bjhw->bihw',matrix,linear)
    adapt=rgb.new_tensor([[1.0479297925449969,.022946870601609652,-.05019226628920524],[.02962780877005599,.9904344267538799,-.017073799063418826],[-.009243040646204504,.015055191490298152,.7518742814281371]])
    xyz=torch.einsum('ij,bjhw->bihw',adapt,xyz)/rgb.new_tensor([.3457/.3585,1.,(1-.3457-.3585)/.3585])[None,:,None,None]
    eps=216/24389;kappa=24389/27
    f=torch.where(xyz>eps,xyz.clamp_min(eps).pow(1/3),(kappa*xyz+16)/116)
    return torch.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],1)


def color_objective(prediction,truth):
    if prediction.shape!=truth.shape:raise ValueError('Different image shape')
    a,b=lab(prediction),lab(truth.detach())
    return .2*F.mse_loss(prediction,truth.detach())+F.mse_loss(a[:,:1]/100,b[:,:1]/100)+5*F.mse_loss(a[:,1:]/128,b[:,1:]/128)
