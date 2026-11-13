"""GT-assisted capacity diagnostics only; never a reference inference engine."""
import numpy as np
import torch
import torch.nn.functional as F


def _rgb(value):
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3 or min(value.shape[:2])<1:raise ValueError('Expected nonempty RGB8')


def block_split(height,width,tile=32):
    if any(type(v)!=int or v<1 for v in [height,width,tile]):raise ValueError('Invalid split dimensions')
    y,x=np.indices((height,width));mask=((y//tile+x//tile)%2==0).ravel()
    if not mask.any() or mask.all():raise ValueError('Need both spatial folds')
    return mask


def polynomial_features(rgb,degree):
    _rgb(rgb)
    if type(degree)!=int or degree not in (1,2):raise ValueError('Expected affine or quadratic')
    x=rgb.reshape(-1,3).astype(np.float64)/255;r,g,b=x.T
    columns=[r,g,b,np.ones(len(x))]
    if degree==2:columns += [r*r,g*g,b*b,r*g,r*b,g*b]
    return np.column_stack(columns)


def ridge_fit(features,truth,train,ridge=1e-5,regions=1):
    x=np.asarray(features,np.float64);y=np.asarray(truth,np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or min(x.shape+y.shape)<1 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid fitting arrays')
    if not isinstance(train,np.ndarray) or train.dtype!=bool or train.shape!=(x.shape[0],) or not train.any() or train.all():raise ValueError('Invalid fitting split')
    if isinstance(ridge,bool) or not np.isfinite(ridge) or ridge<=0 or type(regions)!=int or regions<1:raise ValueError('Invalid regularization')
    selected=x[train];n=len(selected);gram=selected.T@selected/n;gram.flat[::len(gram)+1]+=ridge/regions
    result=np.linalg.solve(gram,selected.T@y[train]/n)
    if not np.isfinite(result).all():raise ValueError('Nonfinite fitted coefficients')
    return result


def rgb_collision_floor(source,truth):
    _rgb(source);_rgb(truth)
    if source.shape!=truth.shape:raise ValueError('RGB collision inputs differ')
    values=source.reshape(-1,3).astype(np.uint32);gt=truth.reshape(-1,3).astype(float);keys=(values[:,0]<<16)|(values[:,1]<<8)|values[:,2]
    unique,indices,counts=np.unique(keys,return_inverse=True,return_counts=True);starts=np.r_[0,np.cumsum(counts)[:-1]];medians=np.empty((len(unique),3))
    for c in range(3):
        ordered=gt[np.lexsort((gt[:,c],indices)),c];medians[:,c]=(ordered[starts+(counts-1)//2]+ordered[starts+counts//2])/2
    error=np.abs(gt-medians[indices]);repeated=counts[indices]>1
    return dict(minimum_mae=float(error.mean()),unique_keys=len(unique),repeated_keys=int((counts>1).sum()),repeated_key_pixels=int(repeated.sum()),
        repeated_pixel_fraction=float(repeated.mean()),repeated_pixel_mae=float(error[repeated].mean()) if repeated.any() else 0.,maximum_key_count=int(counts.max()))


def _tables(model):
    arrays=[model.post_guide._mono(getattr(model.post_guide,name),0.,1.).detach().cpu().numpy().astype(float) for name in ['lut_L','lut_Ct','lut_Cp']]
    result=np.stack(arrays,1)
    if result.shape!=(64,3) or not np.isfinite(result).all() or not (np.diff(result,axis=0)>0).all():raise ValueError('Post guide must be finite and strictly monotonic')
    return result


def inverse_post(rgb,tables):
    _rgb(rgb);flat=rgb.reshape(-1,3).astype(float)/255;domain=np.linspace(-1,1,64)
    values=np.column_stack([np.interp(flat[:,c],tables[:,c],domain) for c in range(3)])
    unreachable=(flat<tables[0])|(flat>tables[-1])
    return values,dict(unreachable_channel_fraction=float(unreachable.mean()),unreachable_pixel_fraction=float(unreachable.any(1).mean()))


def post_forward(raw,tables):
    raw=np.asarray(raw,np.float64)
    if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all():raise ValueError('Invalid matrix output')
    domain=np.linspace(-1,1,64);out=np.column_stack([np.interp(raw[:,c],domain,tables[:,c]) for c in range(3)])
    return np.rint(np.clip(out,0,1)*255).astype(np.uint8)


def _tensor(model,rgb):
    return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).to(next(model.parameters()).device)


@torch.inference_mode()
def native_features(model,api,rgb):
    _rgb(rgb);c=_tensor(model,rgb);h,w=rgb.shape[:2]
    pre=model.pre_guide((c-.5)*2);z=(model.sampler.pre_guide(c)*F.softmax(model.sampler.L_weights,1)).sum(1)
    zero=torch.zeros_like(z);zcoords=torch.stack([zero,zero,(z-.5)*2],-1)[:,None].clamp(-1,1)
    identity=torch.eye(8,device=c.device,dtype=c.dtype).reshape(1,8,8,1,1)
    zw=F.grid_sample(identity,zcoords,align_corners=False,padding_mode='border').squeeze(2)[0].permute(1,2,0).reshape(-1,8).cpu().numpy().astype(float)
    basis=torch.eye(9,device=c.device,dtype=c.dtype).reshape(1,9,3,3);basis=F.interpolate(basis,(16,16),mode='bilinear',align_corners=True)[:,:,None].expand(-1,-1,8,-1,-1).contiguous()
    smoothed=api['lowpass'](basis);alpha=api['lowpass'](torch.ones(1,1,8,16,16,device=c.device,dtype=c.dtype))
    yy,xx=torch.meshgrid((2*torch.arange(h,device=c.device)+1)/h-1,(2*torch.arange(w,device=c.device)+1)/w-1,indexing='ij');coords=torch.stack([xx,yy],-1)[None]
    xy=F.grid_sample(smoothed[:,:,0],coords,align_corners=False,padding_mode='border')/F.grid_sample(alpha[:,:,0],coords,align_corners=False,padding_mode='border')
    xy=xy[0].permute(1,2,0).reshape(-1,9).cpu().numpy().astype(float);inputs=np.column_stack([pre[0].permute(1,2,0).reshape(-1,3).cpu().numpy(),np.ones(h*w)])
    global_features=(zw[:,:,None]*inputs[:,None,:]).reshape(h*w,32);spatial=(xy[:,:,None]*global_features[:,None,:]).reshape(h*w,288)
    if not np.isfinite(global_features).all() or not np.isfinite(spatial).all():raise ValueError('Nonfinite guide features')
    return dict(guided_global=global_features,guided_spatial=spatial,post_tables=_tables(model),guide=dict(z_min=float(z.min()),z_max=float(z.max()),active_z_layers=int((zw>1e-7).any(0).sum()),z_weight_mass=zw.mean(0).tolist()))


@torch.inference_mode()
def render_coefficients(model,api,rgb,coefficients):
    _rgb(rgb);coef=np.asarray(coefficients,np.float64)
    if coef.shape not in [(32,3),(288,3)] or not np.isfinite(coef).all():raise ValueError('Invalid native coefficients')
    regions=coef.shape[0]//32;side=1 if regions==1 else 3;params=coef.reshape(regions,8,4,3)
    matrices=params[:,:,:3,:].transpose(0,1,3,2).reshape(regions,8,9);field=np.concatenate([matrices,params[:,:,3,:]],2)
    c=_tensor(model,rgb);coarse=torch.from_numpy(field.transpose(2,1,0).copy()).reshape(1,96,side,side).to(device=c.device,dtype=c.dtype)
    numerators=F.interpolate(coarse,(16,16),mode='bilinear',align_corners=True).reshape(1,12,8,16,16)
    grid=api['lowpass'](torch.cat([numerators,torch.ones(1,1,8,16,16,device=c.device,dtype=c.dtype)],1))
    output=model.apply_coeffs(model.sampler(grid,c),c)
    if not torch.isfinite(output).all():raise ValueError('Nonfinite native render')
    return np.rint(output[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
