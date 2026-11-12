"""Validated same-layout spatial color field, independent research branch.

Not correspondence across different scenes. A rejected fit supplies no transform.
Grid interpolation is continuous; cross-channel affine terms express saturation.
"""
import numpy as np
from PIL import Image

VERSION='validated-spatial-affine-v1'


def _rgb(a):
    if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<16:
        raise ValueError('Expected RGB8 image with both dimensions >=16')


def _basis(h,w,start=0,end=None):
    end=h if end is None else end
    y=np.arange(start,end,dtype=np.float32)/max(h-1,1)*2
    x=np.arange(w,dtype=np.float32)/max(w-1,1)*2
    by=np.maximum(0,1-np.abs(y[:,None]-np.arange(3)))
    bx=np.maximum(0,1-np.abs(x[:,None]-np.arange(3)))
    return (by[:,None,:,None]*bx[None,:,None,:]).reshape(-1,9)


def _design(rgb,basis):
    x=np.column_stack([rgb.reshape(-1,3).astype(float)/255,np.ones(rgb.shape[0]*rgb.shape[1])])
    return (basis[:,:,None]*x[:,None,:]).reshape(-1,36)


def fit(source,reference):
    _rgb(source);_rgb(reference)
    rejected=dict(accepted=False,version=VERSION)
    if source.shape!=reference.shape:
        return dict(rejected,reason='Different layout dimensions; no correspondence established')
    h,w=source.shape[:2]
    size=(max(16,round(w*min(1,256/max(h,w)))),max(16,round(h*min(1,256/max(h,w)))))
    a=np.asarray(Image.fromarray(source).resize(size,Image.Resampling.BILINEAR))
    b=np.asarray(Image.fromarray(reference).resize(size,Image.Resampling.BILINEAR))
    ya=a.astype(float) @ [.299,.587,.114]; yb=b.astype(float) @ [.299,.587,.114]
    ga=np.concatenate([np.diff(ya,axis=0).ravel(),np.diff(ya,axis=1).ravel()])
    gb=np.concatenate([np.diff(yb,axis=0).ravel(),np.diff(yb,axis=1).ravel()])
    if min(ga.std(),gb.std())<1:
        return dict(rejected,reason='Insufficient structure')
    correlation=float(np.corrcoef(ga,gb)[0,1])
    rejected.update(gradient_correlation=correlation)
    if correlation<.9:
        return dict(rejected,reason='Insufficient same-layout structure')
    hh,ww=a.shape[:2]
    yy,xx=np.indices((hh,ww)); train=((xx//8+yy//8)%2==0).ravel()
    flat=a.reshape(-1,3).astype(float)/255
    x=np.column_stack([flat,np.ones(len(flat))]); y=b.reshape(-1,3).astype(float)/255
    if np.linalg.cond(x[train])>200:
        return dict(rejected,reason='Insufficient color diversity')
    matrix=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
    globalerror=np.abs(np.clip(x[~train]@matrix,0,1)-y[~train])*255
    rejected['global_holdout_mae']=float(globalerror.mean())
    if globalerror.mean()<=1.5 and np.quantile(globalerror,.95)<=4:
        return dict(rejected,reason='Global affine already adequate; defer to global path')
    design=_design(a,_basis(hh,ww))
    coefficients=[]
    for c in range(3):
        usable=train&(y[:,c]>2/255)&(y[:,c]<253/255)
        if usable.sum()<256:
            return dict(rejected,reason='Insufficient unclipped samples')
        # Fixed ridge stabilizes spatial field coefficients; holdout is untouched.
        d=design[usable]
        coefficients.append(np.linalg.solve(d.T@d+np.eye(36)*.001,d.T@y[usable,c]))
    coefficients=np.stack(coefficients,axis=1)
    predicted=design@coefficients
    error=np.abs(np.clip(predicted,0,1)-y)*255
    pixelerror=error.mean(axis=1).reshape(hh,ww)
    tiles=[float(pixelerror[j:j+8,i:i+8].mean()) for j in range(0,hh,8) for i in range(0,ww,8)]
    rejected.update(holdout_mae=float(error[~train].mean()),holdout_p95=float(np.quantile(error[~train],.95)),
                    max_tile_mae=max(tiles))
    if rejected['holdout_mae']>1.5 or rejected['holdout_p95']>4 or max(tiles)>6:
        return dict(rejected,reason='Spatial color prediction fails independent holdout')
    # Check transforms over the entire RGB unit cube at every grid node.
    nodes=coefficients.reshape(9,4,3)
    if np.max(np.abs(nodes[:,3]))>.3 or np.max(np.sum(np.abs(nodes[:,:3]),axis=1))>3:
        return dict(rejected,reason='Unsupported color transform amplitude')
    return dict(rejected,accepted=True,coefficients=coefficients.tolist(),
                limitation='Verified same-layout color relation only; not arbitrary scene style understanding')


def apply(rgb,model,strength=1.):
    _rgb(rgb)
    if isinstance(strength,(bool,np.bool_)) or not np.isscalar(strength) or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Invalid strength')
    if model.get('version')!=VERSION or not model.get('accepted'):
        raise ValueError('Unvalidated spatial transform')
    coefficients=np.asarray(model['coefficients'],dtype=float)
    if coefficients.shape!=(36,3) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid spatial coefficients')
    h,w=rgb.shape[:2]; result=np.empty_like(rgb)
    # Interpolate the 3x3 coefficient grid directly instead of materializing
    # 36 regression features per output pixel. Fit remains unchanged.
    nodes=coefficients.reshape(3,3,4,3).astype(np.float32)
    xpos=np.arange(w,dtype=np.float32)/max(w-1,1)*2
    bx=np.maximum(0,1-np.abs(xpos[:,None]-np.arange(3,dtype=np.float32)))
    horizontal=np.einsum('wi,yico->ywco',bx,nodes)
    for start in range(0,h,32):
        end=min(h,start+32)
        x=rgb[start:end]
        ypos=np.arange(start,end,dtype=np.float32)/max(h-1,1)*2
        by=np.maximum(0,1-np.abs(ypos[:,None]-np.arange(3,dtype=np.float32)))
        field=np.einsum('hy,ywco->hwco',by,horizontal)
        transformed=np.einsum('hwc,hwco->hwo',x.astype(np.float32),field[:,:,:3,:])+field[:,:,3,:]*255
        result[start:end]=np.rint(np.clip(x+strength*(transformed-x),0,255)).astype(np.uint8)
    return result
