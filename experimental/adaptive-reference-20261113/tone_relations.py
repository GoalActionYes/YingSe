"""Training-only tone response constraints with masks fixed by target content."""
import torch
import torch.nn.functional as F


def _rgb(value):
    if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=3 or min(value.shape)<1 or value.dtype not in (torch.float32,torch.float64) or not torch.isfinite(value).all() or (value<0).any() or (value>1).any():
        raise ValueError('Expected finite floating RGB BCHW within [0,1]')


def source_tone_weights(content):
    _rgb(content);source=content.detach();y=(source*source.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    centers=source.new_tensor([0.,.5,1.])[None,:,None,None];soft=(1-(y-centers).abs()*2).clamp(0,1)
    return torch.cat([torch.ones_like(y),soft],1).detach()


def tone_descriptor(rgb,weights):
    _rgb(rgb)
    if not isinstance(weights,torch.Tensor) or weights.shape!=(rgb.shape[0],4,*rgb.shape[2:]) or weights.device!=rgb.device or weights.dtype!=rgb.dtype or weights.requires_grad or not torch.isfinite(weights).all() or (weights<0).any() or (weights>1).any():raise ValueError('Invalid fixed source weights')
    matrix=rgb.new_tensor([[.2126,.7152,.0722],[-.1146,-.3854,.5],[.5,-.4542,-.0458]])
    values=torch.einsum('bchw,kc->bkhw',rgb,matrix);y,cb,cr=values.unbind(1);span=rgb.max(1).values-rgb.min(1).values
    fields=torch.stack([y,y.square(),cb,cr,span],1);global_mean=fields.mean((-2,-1))[:,None,:]
    sums=torch.einsum('brhw,bchw->brc',weights,fields)/(rgb.shape[2]*rgb.shape[3]);mass=weights.mean((-2,-1))[:,:,None]
    moments=(sums+.01*global_mean)/(mass+.01)
    spread=((moments[:,:,1]-moments[:,:,0].square()).clamp_min(0)+1e-4).sqrt()
    result=torch.stack([moments[:,:,0],spread,moments[:,:,2],moments[:,:,3],moments[:,:,4]],-1)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite tone descriptor')
    return result


def tone_relation_losses(prediction,truth,content):
    for value in [prediction,truth,content]:_rgb(value)
    if prediction.shape!=truth.shape or prediction.shape!=content.shape or prediction.shape[0]!=4 or any(v.device!=prediction.device or v.dtype!=prediction.dtype for v in [truth,content]):raise ValueError('Expected two RGB reference pairs with matching dimensions/device/dtype')
    if not torch.equal(content[::2],content[1::2]):raise ValueError('Reference pairs must share identical target content')
    fixed=source_tone_weights(content);pred=tone_descriptor(prediction,fixed);gt=tone_descriptor(truth.detach(),fixed);scale=prediction.new_tensor([1,1,2,2,1])
    return F.smooth_l1_loss(pred*scale,gt*scale,beta=1/255),F.smooth_l1_loss((pred[1::2]-pred[::2])*scale,(gt[1::2]-gt[::2])*scale,beta=1/255)
