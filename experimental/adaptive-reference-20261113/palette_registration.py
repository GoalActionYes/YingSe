"""Independent research: trimmed bidirectional palette ICP, not semantic grading."""
import numpy as np


def palette(rgb):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('RGB8 expected')
    x=rgb.reshape(-1,3);keys=(x.astype(np.int32)//16)@np.array([256,16,1])
    _,inverse,count=np.unique(keys,return_inverse=True,return_counts=True)
    centers=np.stack([np.bincount(inverse,weights=x[:,i],minlength=len(count)) for i in range(3)],1)/count[:,None]/255
    # Bound matching cost; deterministic farthest-point coverage, no area labels.
    if len(centers)>256:
        selected=[int(np.argmin(np.sum((centers-.5)**2,1)))];distance=np.full(len(centers),np.inf)
        for _ in range(255):
            distance=np.minimum(distance,np.sum((centers-centers[selected[-1]])**2,1));selected.append(int(np.argmax(distance)))
        centers=centers[selected]
    return centers


def register(source,reference):
    x,y=np.asarray(source,dtype=float),np.asarray(reference,dtype=float)
    if any(a.ndim!=2 or a.shape[1]!=3 or len(a)<4 or not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in [x,y]):
        raise ValueError('Expected at least four finite unit RGB palette points')
    design=np.c_[x,np.ones(len(x))];identity=np.r_[np.eye(3),np.zeros((1,3))];matrix=identity.copy()
    history=[]
    for iteration in range(20):
        mapped=design@matrix;dist=np.sum((mapped[:,None]-y[None])**2,2)
        forward=np.argmin(dist,1);reverse=np.argmin(dist,0)
        indices=np.r_[np.arange(len(x)),reverse];targets=np.r_[forward,np.arange(len(y))]
        errors=dist[indices,targets];keep=np.argsort(errors,kind='stable')[:max(4,int(.8*len(errors)))];a=design[indices[keep]];b=y[targets[keep]]
        scale=max(float(np.median(errors[keep])),1e-6);weight=1/(1+errors[keep]/scale)
        reg=np.diag([1e-4,1e-4,1e-4,1e-4])
        updated=np.linalg.solve(a.T@(a*weight[:,None])+reg,a.T@(b*weight[:,None])+reg@identity)
        u,s,v=np.linalg.svd(updated[:3]);updated[:3]=(u*np.clip(s,.5,2))@v;updated[3]=np.clip(updated[3],-.2,.2)
        change=float(np.max(np.abs(updated-matrix)));matrix=updated;history.append(dict(trimmed_mse=float(errors[keep].mean()),change=change))
        if change<1e-7:break
    return matrix,dict(iterations=len(history),history=history,source_points=len(x),reference_points=len(y),
        limitation='Nearest colors are not verified corresponding materials; global affine only')


def apply(rgb,matrix):
    values=rgb.astype(np.float32)/255
    return np.rint(np.clip(values@matrix[:3]+matrix[3],0,1)*255).astype(np.uint8)
