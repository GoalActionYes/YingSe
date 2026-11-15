"""Research-only frozen-backbone condition head, no inference-time GT inputs."""
import torch
import torch.nn.functional as F
from training_checks import finite_output


class ConditionHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(3072,128),torch.nn.GELU(),torch.nn.Linear(128,128))
        torch.nn.init.zeros_(self.layers[-1].weight);torch.nn.init.zeros_(self.layers[-1].bias)

    def forward(self,content,reference):
        for value in [content,reference]:
            if not isinstance(value,torch.Tensor) or value.ndim!=2 or value.shape[1]!=1024 or value.shape[0]<1 or not value.is_floating_point() or not torch.isfinite(value).all() or (value.norm(dim=1)<=1e-12).any():raise ValueError('Expected finite nonzero B1024 features')
        if content.shape!=reference.shape or content.device!=reference.device or content.dtype!=reference.dtype:raise ValueError('Feature batches differ')
        a,b=[F.normalize(v.detach(),dim=1) for v in [content,reference]]
        return finite_output(self.layers(torch.cat([a,b,b-a],1)),(a.shape[0],128))


def freeze_and_create(model):
    model.eval().requires_grad_(False);p=next(model.parameters())
    return ConditionHead().to(device=p.device,dtype=p.dtype)


def condition_forward(model,head,api,content,reference,zc=None,zr=None,enabled=True):
    if type(enabled)!=bool:raise ValueError('Invalid condition enable flag')
    hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](content));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](reference))
    if not enabled:return model(content,hc,hs)
    low=api['TF'].resize(content,(64,64),antialias=True)
    img=model.img_projection(low);context=model.hist_projection(torch.cat([hc,hs],1))
    delta=head(zc,zr)
    if context.shape!=(content.shape[0],128,1):raise ValueError('Unexpected author context shape')
    context=context+delta[:,:,None]
    raw=model.fusion(torch.cat([img,model.attention(img,context)],1)).reshape(-1,13,8,16,16)
    alpha=F.softplus(raw[:,-1:])+1e-3
    grid=api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1))
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),finite_output(grid,(content.shape[0],13,8,16,16))
