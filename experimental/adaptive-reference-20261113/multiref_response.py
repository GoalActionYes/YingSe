"""Measure paired reference response at fixed content, using RGB code values."""
import numpy as np


def response_metrics(prediction_a,prediction_b,gt_a,gt_b):
    arrays=[np.asarray(a,dtype=np.float64) for a in [prediction_a,prediction_b,gt_a,gt_b]]
    shape=arrays[0].shape
    if len(shape)!=3 or shape[-1]!=3 or any(n<1 for n in shape):raise ValueError('Expected nonempty HWC RGB')
    if any(a.shape!=shape or not np.isfinite(a).all() or (a<0).any() or (a>255).any() for a in arrays):raise ValueError('Invalid paired RGB inputs')
    pa,pb,ga,gb=arrays;actual=pb-pa;expected=gb-ga;energy=float(np.mean(expected**2))
    return dict(gt_change_mae=float(np.abs(expected).mean()),prediction_change_mae=float(np.abs(actual).mean()),
        delta_mae=float(np.abs(actual-expected).mean()),response_gain=float(np.mean(actual*expected)/energy) if energy>0 else None)
