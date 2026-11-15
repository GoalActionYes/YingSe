"""Fixed bounded five-control per-pair solve; no model weight optimization."""
import sys
import numpy as np
from independent_coco import ROOT
from spatial_render import validate_rgb
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from reference_color_app.adjustments import transform

AXES=('brightness','contrast','saturation','temperature','tint')


def retouch(rgb,u):
    validate_rgb(rgb);u=np.asarray(u,dtype=np.float64)
    if u.shape!=(5,) or not np.isfinite(u).all() or (np.abs(u)>1).any():raise ValueError('Expected five finite bounded control parameters')
    if not u.any():return rgb.copy()
    state={'basic':dict(zip(AXES,(u*20).tolist()))}
    return np.rint(transform(rgb.astype(np.float32)/255,state).clip(0,1)*255).astype(np.uint8)


def normalized(value):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or min(value.shape)<1 or not np.isfinite(value).all():raise ValueError('Expected finite feature matrix')
    norms=np.linalg.norm(value,axis=1,keepdims=True)
    if (norms<=1e-12).any():raise ValueError('Zero feature vector')
    return value/norms


def solve(features,reference):
    values=normalized(features);ref=normalized(np.asarray(reference)[None])[0]
    if values.shape!=(11,len(ref)):raise ValueError('Expected baseline and five central-difference pairs')
    j=((values[1::2]-values[2::2])/2).T;gram=j.T@j;trace=float(np.trace(gram))
    if trace<=1e-12:return np.zeros(5)
    return np.linalg.solve(gram+np.eye(5)*(.1*trace/5),j.T@(ref-values[0])).clip(-1,1)


def select(distances,endpoints):
    distances=np.asarray(distances,dtype=np.float64);endpoints=np.asarray(endpoints,dtype=np.float64)
    if distances.shape!=(4,) or endpoints.shape!=(4,) or not np.isfinite(distances).all() or not np.isfinite(endpoints).all() or (distances<0).any() or (distances>2).any() or (endpoints<0).any() or (endpoints>1).any():raise ValueError('Expected four finite cosine distances and endpoint fractions')
    eligible=(distances<distances[0]-.001)&(endpoints<=endpoints[0]+.005);eligible[0]=True
    return int(np.argmin(np.where(eligible,distances,np.inf)))
