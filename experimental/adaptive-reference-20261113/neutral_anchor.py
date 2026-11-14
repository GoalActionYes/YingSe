"""Unintegrated neutral-highlight diagnostic. Fixed parameters; no semantics."""
import numpy as np

LUMA=np.array([.299,.587,.114])


def _rgb(value):
    if not isinstance(value,np.ndarray) or value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3 or min(value.shape[:2])<1:
        raise ValueError('Expected nonempty RGB8')
    return value.astype(float)/255


def _anchors(x):
    y=x@LUMA;spread=x.max(-1)-x.min(-1)
    mask=(y>.65)&(spread<.12)&(x.max(-1)<.98)
    return y,spread,mask


def correct_neutral_highlights(source,reference,graded):
    s,r,g=map(_rgb,(source,reference,graded))
    if s.shape!=g.shape:raise ValueError('Source/output dimensions differ')
    y,spread,sm=_anchors(s);ry,_,rm=_anchors(r)
    info=dict(source_anchor_pixels=int(sm.sum()),reference_anchor_pixels=int(rm.sum()),bypass=True)
    if sm.sum()<max(64,.005*sm.size) or rm.sum()<max(64,.005*rm.size):
        return graded.copy(),dict(info,reason='Insufficient neutral bright support')
    ref=np.median((r-ry[...,None])[rm],axis=0);gy=g@LUMA
    actual=np.median((g-gy[...,None])[sm],axis=0)
    delta=ref-actual;delta=delta-float(delta@LUMA)
    delta*=min(1.,.04/max(float(np.max(np.abs(delta))),1e-12))*.5
    weight=np.clip((y-.55)/.25,0,1)*np.clip(1-spread/.12,0,1)
    output=np.rint(np.clip(g+weight[...,None]*delta,0,1)*255).astype(np.uint8)
    return output,dict(info,bypass=False,delta=delta.tolist(),reference_neutral_chroma=ref.tolist(),predicted_neutral_chroma=actual.tolist(),affected_pixels=int((weight>0).sum()))
