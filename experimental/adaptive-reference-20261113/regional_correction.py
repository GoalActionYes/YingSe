"""Zero-initialized global plus soft-region affine residual; research only."""
import torch


def finite(value,shape):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shape) or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('Invalid correction tensor')


def global_control(descriptors):
    if descriptors.ndim!=2:raise ValueError('Expected descriptor batch')
    finite(descriptors,(descriptors.shape[0],1154));result=descriptors.clone()
    result[:,384:768]=descriptors[:,:384];result[:,768:1152]=descriptors[:,:384]
    return result


def apply_residual(rgb,baseline,probability,coefficients):
    if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('Expected BCHW RGB')
    b,_,h,w=rgb.shape;finite(rgb,(b,3,h,w));finite(baseline,rgb.shape);finite(probability,(b,1,h,w));finite(coefficients,(b,3,3,4))
    if (probability<0).any() or (probability>1).any():raise ValueError('Invalid person probability')
    x=torch.cat([rgb,torch.ones_like(rgb[:,:1])],1)
    parts=torch.einsum('bkij,bjhw->bkihw',coefficients,x)
    residual=parts[:,0]+probability*parts[:,1]+(1-probability)*parts[:,2]
    result=baseline+.1*torch.tanh(residual)
    finite(result,baseline.shape);return result


class RegionalCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(3462,128),torch.nn.GELU(),torch.nn.Linear(128,36))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self,rgb,baseline,probability,content,reference,enabled=True):
        if type(enabled)!=bool:raise ValueError('Invalid enable flag')
        for value in [content,reference]:finite(value,(rgb.shape[0],1154))
        if not enabled:return baseline
        features=torch.cat([content,reference,reference-content],1)
        coefficients=self.layers(features).reshape(-1,3,3,4)
        return apply_residual(rgb,baseline,probability,coefficients)
