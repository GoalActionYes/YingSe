"""RGB-only fixed soft-person background chroma removal, research intervention."""
import numpy as np

def normalize(rgb,probability):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('Expected RGB8')
    if not isinstance(probability,np.ndarray) or probability.shape!=rgb.shape[:2] or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():
        raise ValueError('Expected automatic probability in [0,1]')
    x=rgb.astype(np.float64);p=probability.astype(np.float64)[...,None]
    y=(x@np.array([.2126,.7152,.0722]))[...,None]
    result=p*x+(1-p)*y
    return np.rint(result.clip(0,255)).astype(np.uint8)
