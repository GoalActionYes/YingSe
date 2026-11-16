"""Isolate reference Y and CbCr histograms; diagnostic only."""
import numpy as np


def conditions(original,changed):
    for value in (original,changed):
        if not isinstance(value,np.ndarray) or value.shape!=(1,3,128) or not np.isfinite(value).all() or (value<0).any() or not np.allclose(value.sum(-1),1,rtol=0,atol=1e-6):
            raise ValueError('Expected normalized nonnegative finite 1x3x128 histogram')
    y=original.copy();y[:,:1]=changed[:,:1]
    chroma=original.copy();chroma[:,1:]=changed[:,1:]
    return dict(full=changed.copy(),y_only=y,chroma_only=chroma)
