"""Exact RGB8 point-function diagnostic; not a LUT estimator for deployment."""
import numpy as np
from capacity_audit import rgb_collision_floor


def keys(rgb):
    v=rgb.reshape(-1,3).astype(np.uint32)
    return (v[:,0]<<16)|(v[:,1]<<8)|v[:,2]


def analyze(source_a,result_a,source_b,result_b):
    for x,y in [(source_a,result_a),(source_b,result_b)]:
        if not isinstance(x,np.ndarray) or not isinstance(y,np.ndarray) or x.dtype!=np.uint8 or y.dtype!=np.uint8 or x.shape!=y.shape or x.ndim!=3 or x.shape[2]!=3 or min(x.shape[:2])<1:raise ValueError('Expected same-size RGB8 input/output pairs')
    # Flatten each image to width1 so source image aspect ratios need not match.
    x=np.concatenate([source_a.reshape(-1,1,3),source_b.reshape(-1,1,3)])
    y=np.concatenate([result_a.reshape(-1,1,3),result_b.reshape(-1,1,3)])
    joint=rgb_collision_floor(x,y);a=rgb_collision_floor(source_a,result_a);b=rgb_collision_floor(source_b,result_b)
    na=source_a.size//3;nb=source_b.size//3
    within=(a['minimum_mae']*na+b['minimum_mae']*nb)/(na+nb)
    assert joint['minimum_mae']>=within-1e-10
    ka,kb=keys(source_a),keys(source_b);ua,ia,ca=np.unique(ka,return_inverse=True,return_counts=True);ub,ib,cb=np.unique(kb,return_inverse=True,return_counts=True)
    shared=np.intersect1d(ua,ub)
    def median(values,inverse,counts):
        v=values.reshape(-1,3);starts=np.r_[0,np.cumsum(counts)[:-1]];out=[]
        for c in range(3):
            ordered=v[np.lexsort((v[:,c],inverse)),c].astype(float)
            out.append((ordered[starts+(counts-1)//2]+ordered[starts+counts//2])/2)
        return np.stack(out,axis=1)
    ma=median(result_a,ia,ca);mb=median(result_b,ib,cb)
    gap=float(np.abs(ma[np.searchsorted(ua,shared)]-mb[np.searchsorted(ub,shared)]).mean()) if len(shared) else None
    return dict(joint_minimum_mae=joint['minimum_mae'],reference_minimum_mae=a['minimum_mae'],target_minimum_mae=b['minimum_mae'],within_weighted_minimum_mae=float(within),
        cross_image_excess_minimum_mae=float(max(0,joint['minimum_mae']-within)),shared_rgb_keys=len(shared),
        reference_shared_pixel_fraction=float(np.isin(ka,shared).mean()),target_shared_pixel_fraction=float(np.isin(kb,shared).mean()),shared_key_median_output_gap=gap)
