"""Descriptive same-content reference response; not semantic or perceptual truth."""
import numpy as np


def tone_signature(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Expected nonempty RGB8 image')
    x=rgb.astype(np.float64);y=x@np.array([.299,.587,.114]);cb=(x[...,2]-y)/1.772;cr=(x[...,0]-y)/1.402
    high=y>np.median(y);low=~high
    def difference(chroma):return float(chroma[high].mean()-chroma[low].mean()) if high.any() and low.any() else 0.
    return dict(brightness=float(y.mean()),contrast=float(y.std()),saturation=float((x.max(-1)-x.min(-1)).mean()),
        cb=float(cb.mean()),cr=float(cr.mean()),light_minus_dark_cb=difference(cb),light_minus_dark_cr=difference(cr))


def tone_response(prediction0,prediction1,truth0,truth1):
    values=[tone_signature(x) for x in [prediction0,prediction1,truth0,truth1]]
    if len({x.shape for x in [prediction0,prediction1,truth0,truth1]})!=1:raise ValueError('Response image dimensions differ')
    p={key:values[1][key]-values[0][key] for key in values[0]};g={key:values[3][key]-values[2][key] for key in values[0]}
    return dict(prediction_delta=p,gt_delta=g,absolute_delta_error={k:abs(p[k]-g[k]) for k in p},
        direction_correct={k:bool(np.sign(p[k])==np.sign(g[k])) if abs(g[k])>=2 else None for k in p})
