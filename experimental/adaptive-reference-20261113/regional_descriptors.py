"""RGB-only regional marginals; preparation for a separately trained residual head."""
import numpy as np

YCC=np.array([[.2126,.7152,.0722],[-.1146,-.3854,.5],[.5,-.4542,-.0458]],np.float32)


def validate(rgb,p):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:raise ValueError('Expected nonempty RGB8')
    if not isinstance(p,np.ndarray) or p.shape!=rgb.shape[:2] or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Expected same-size probability in [0,1]')


def histogram(rgb,weights):
    validate(rgb,weights)
    x=(rgb.astype(np.float32)/np.float32(255))@YCC.T
    x[...,0]=np.clip(x[...,0],0,1);x[...,1:]=np.clip(x[...,1:],-.5,.5)+np.float32(.5)
    bins=np.floor(x*128).clip(0,127).astype(np.int64).reshape(-1,3);w=weights.ravel().astype(np.float64)
    total=w.sum()
    if total==0:return np.zeros((3,128),np.float32)
    return np.stack([np.bincount(bins[:,i],weights=w,minlength=128)/total for i in range(3)]).astype(np.float32)


def describe(rgb,probability):
    validate(rgb,probability);p=probability.astype(np.float32)
    return np.concatenate([histogram(rgb,np.ones(p.shape,np.float32)).ravel(),histogram(rgb,p).ravel(),
        histogram(rgb,1-p).ravel(),np.array([p.mean(),(p>.5).mean()],np.float32)])
