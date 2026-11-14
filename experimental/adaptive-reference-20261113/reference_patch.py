"""Reference feature-set conditioning; no semantic or geometric correspondence claim."""
import torch
import torch.nn.functional as F
from training_checks import finite_output
from pairwise_gate import response_development_gate


def reference_tokens(features,pooled=False):
    if type(pooled)!=bool or not isinstance(features,torch.Tensor) or features.ndim!=4 or features.shape[1]!=128 or min(features.shape)<1 or not features.is_floating_point() or not torch.isfinite(features).all():raise ValueError('Expected finite B128HW features and boolean mode')
    tokens=F.adaptive_avg_pool2d(features,(4,4)).flatten(2).transpose(1,2)
    if pooled:tokens=tokens.mean(1,keepdim=True).expand(-1,16,-1)
    return tokens


class ReferencePatchContext(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Linear(128,128,bias=False);self.kv=torch.nn.Linear(128,256,bias=False);self.proj=torch.nn.Linear(128,128)
        torch.nn.init.zeros_(self.proj.weight);torch.nn.init.zeros_(self.proj.bias)

    def forward(self,query,tokens):
        if not isinstance(query,torch.Tensor) or query.ndim!=4 or query.shape[1]!=128 or min(query.shape)<1:raise ValueError('Expected nonempty B128HW query')
        if not isinstance(tokens,torch.Tensor) or tokens.shape!=(query.shape[0],16,128):raise ValueError('Expected matching B16x128 tokens')
        for value in [query,tokens]:
            if not value.is_floating_point() or value.dtype!=self.q.weight.dtype or value.device!=self.q.weight.device or not torch.isfinite(value).all():raise ValueError('Feature dtype/device/range differs')
        b,_,h,w=query.shape;q=self.q(query.flatten(2).transpose(1,2)).reshape(b,h*w,8,16).permute(0,2,1,3);kv=self.kv(tokens).reshape(b,16,2,8,16).permute(2,0,3,1,4);k,v=kv[0],kv[1]
        weights=((q@k.transpose(-2,-1))*.25).softmax(-1);values=(weights@v).transpose(1,2).reshape(b,h*w,128)
        output=self.proj(values).transpose(1,2).reshape(b,128,h,w)
        return finite_output(output,query.shape)


def enable_reference_patch(model):
    if any(hasattr(model,name) for name in ['reference_patch','pairwise_adapter','reference_adapter']):raise ValueError('Expected original unextended base model')
    parameter=next(model.parameters());model.reference_patch=ReferencePatchContext().to(device=parameter.device,dtype=parameter.dtype)


def initialize_patch_model(model,base_state):
    if any(hasattr(model,name) for name in ['reference_patch','pairwise_adapter','reference_adapter']):raise ValueError('Expected original unextended base model')
    model.load_state_dict(base_state,strict=True);torch.manual_seed(20261126);enable_reference_patch(model)


def patch_grid(model,api,content,reference,pooled=False,disabled=False):
    if type(pooled)!=bool or type(disabled)!=bool:raise ValueError('Expected boolean patch modes')
    parameter=next(model.parameters())
    for value in [content,reference]:
        if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[1]!=3 or min(value.shape)<1 or value.shape[0]!=content.shape[0] or value.dtype!=parameter.dtype or value.device!=parameter.device or not torch.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Expected matching finite unit RGB batches on model device')
    c,s=api['srgb_to_ycbcr'](content),api['srgb_to_ycbcr'](reference);hc,hs=api['batch_histogram_1D'](c),api['batch_histogram_1D'](s)
    source=model.img_projection(api['TF'].resize(content,(64,64),antialias=True));context=model.hist_projection(torch.cat([hc,hs],1));global_features=model.attention(source,context)
    if not disabled:
        ref_features=model.img_projection(api['TF'].resize(reference,(64,64),antialias=True));global_features=global_features+model.reference_patch(source,reference_tokens(ref_features,pooled))
    raw=model.fusion(torch.cat([source,global_features],1)).reshape(-1,13,8,16,16);alpha=F.softplus(raw[:,-1:])+1e-3
    return finite_output(api['lowpass'](torch.cat([raw[:,:-1]*alpha,alpha],1)),(content.shape[0],13,8,16,16))


def patch_forward(model,api,content,reference,pooled=False,disabled=False):
    grid=patch_grid(model,api,content,reference,pooled,disabled)
    return finite_output(model.apply_coeffs(model.sampler(grid,content),content),content.shape),grid


def patch_development_gate(baseline,selected,mean_control,c10):
    matched=response_development_gate(baseline,selected,mean_control);external=response_development_gate(baseline,selected,c10)
    return dict(matched_mean_control=matched,external_C10=external,proceed_to_independent_evaluation=bool(matched['proceed_to_independent_evaluation'] and external['proceed_to_independent_evaluation']))
