"""Descriptive paired RGB vector geometry, not semantic style correspondence."""
import numpy as np


def response_geometry(prediction0,prediction1,truth0,truth1):
    arrays=[np.asarray(x) for x in (prediction0,prediction1,truth0,truth1)];shape=arrays[0].shape
    if len(shape)!=3 or shape[-1]!=3 or min(shape)<1 or any(x.shape!=shape or x.dtype!=np.uint8 for x in arrays):raise ValueError('Expected equally sized nonempty RGB8 frames')
    p0,p1,g0,g1=[x.astype(np.float64) for x in arrays];actual=p1-p0;expected=g1-g0;energy=float(np.mean(expected**2));actual_energy=float(np.mean(actual**2));dot=float(np.mean(actual*expected))
    if energy==0:return dict(gain=None,amplitude_ratio=None,cosine=None,orthogonal_ratio=None)
    gain=dot/energy
    return dict(gain=gain,amplitude_ratio=float(np.sqrt(actual_energy/energy)),cosine=float(np.clip(dot/np.sqrt(actual_energy*energy),-1,1)) if actual_energy>0 else None,orthogonal_ratio=float(np.sqrt(np.mean((actual-gain*expected)**2)/energy)))
