"""Research: bounded reference skin chroma residual, shared graded luminance.

Caller supplies original-image semantic masks and visible-face eligibility.
Not Jianying logic; not original skin protection; no CUBE representation.
"""
import numpy as np

def match_skin(graded,reference,target_mask,reference_mask):
 masks=[]
 for rgb,m in ((graded,target_mask),(reference,reference_mask)):
  if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
   raise ValueError('Expected RGB8')
  m=np.asarray(m)
  if m.shape!=rgb.shape[:2] or not np.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('Invalid skin mask')
  masks.append(m)
 tm,rm=masks;selected=[tm>.8,rm>.8]
 if min(int(x.sum()) for x in selected)<16:return graded.copy(),dict(applied=False,reason='Insufficient reliable skin')
 y=np.array([.299,.587,.114]);means=[]
 for rgb,s in zip((graded,reference),selected):
  pixels=rgb[s].astype(float);means.append((pixels-(pixels@y)[:,None]).mean(axis=0))
 delta=.5*(means[1]-means[0]);peak=np.abs(delta).max()
 if peak>8:delta*=8/peak
 out=_apply_fixed(graded,tm,delta)
 return out,dict(applied=bool(np.any(out!=graded)),delta_rgb=delta.tolist(),
  max_rgb_delta=8,reference_pull=.5,spatial=True,luminance='graded luminance preserved before 8bit rounding')

def _apply_fixed(graded,mask,delta):
 """Internal video helper; reuse the first-frame fitted delta with moving masks."""
 a=graded.astype(float);change=mask[...,None]*delta
 # Shrink the whole chroma vector near gamut boundaries, preserving luminance.
 ratios=np.ones_like(change)
 np.divide(255-a,change,out=ratios,where=change>0)
 np.divide(-a,change,out=ratios,where=change<0)
 scale=np.clip(ratios.min(axis=2),0,1)
 out=np.rint(a+change*scale[...,None]).clip(0,255).astype(np.uint8)
 return out
