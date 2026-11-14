"""Measure strict pre-clamp channel excursions; not a quality/causality gate."""
import numpy as np


def clipping_statistics(raw,truth):
    x=np.asarray(raw,dtype=np.float64);y=np.asarray(truth,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=3 or x.shape[-1]!=3 or min(x.shape)<1 or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('Expected finite matching HWC RGB and normalized truth')
    outside=(x<0)|(x>1);error=np.abs(np.clip(x,0,1)-y);total=float(error.sum())
    return dict(outside_fraction=float(outside.mean()),outside_error_fraction=float(error[outside].sum()/total) if total>0 else 0.,clipped_l1=float(error.mean()),raw_l1=float(np.abs(x-y).mean()))
