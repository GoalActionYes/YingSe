"""Fixed identity-residual inference diagnostic, not integrated product logic."""
import numpy as np


def cancel(source,raw,self_raw):
    if not isinstance(source,np.ndarray) or source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3 or not source.size:
        raise ValueError('Expected nonempty RGB8')
    for value in (raw,self_raw):
        if not isinstance(value,np.ndarray) or value.shape!=source.shape or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all():
            raise ValueError('Expected finite matching RGB float model output')
    result=source.astype(np.float32)/255+raw-self_raw
    return np.rint(np.clip(result,0,1)*255).astype(np.uint8)
