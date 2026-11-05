"""Research only: capped semantic contribution sampling for ONE shared map.

The zero setting is the same-sampling control, not byte-equivalence with the
shipped image-resize/sparse-feature path. No UI or shipped algorithm is changed.
"""
import numpy as np

VERSION='capped-contribution-sampling-v1'

def balanced_share(p,amount,*,goal=.5):
 if isinstance(amount,(bool,np.bool_)) or not np.isfinite(amount) or amount not in (0,.25,.5):
  raise ValueError('Research amount must be 0, .25 or .5')
 if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid region share')
 if isinstance(goal,(bool,np.bool_)) or not np.isfinite(goal) or not 0<=goal<=1:raise ValueError('Invalid goal share')
 if amount==0:return p
 # Each group's contribution relative to original occupancy stays within [1/2,2].
 lower=max(.5*p,1-2*(1-p));upper=min(2*p,1-.5*(1-p))
 return float(np.clip(p+amount*(goal-p),lower,upper))

def sample_atlas(rgb,mask,amount,side=320):
 balanced_share(.5,amount)
 if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
  raise ValueError('Expected RGB8 image')
 if not isinstance(side,int) or isinstance(side,bool) or side<8:raise ValueError('Invalid side')
 flat=rgb.reshape(-1,3);n=side*side;rng=np.random.default_rng(1105)
 if not len(flat):raise ValueError('Empty image')
 fg=bg=None
 if mask is not None:
  m=np.asarray(mask)
  if m.shape!=rgb.shape[:2] or not np.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('Invalid mask')
  fg=flat[m.ravel()>.8];bg=flat[m.ravel()<.2]
 if fg is None or min(len(fg),len(bg))<max(16,len(flat)*.01) or len(fg)+len(bg)<.5*len(flat):
  a=flat[rng.integers(0,len(flat),n)].reshape(side,side,3)
  return a,dict(applied=False,reason='Missing or insufficient reliable regions',version=VERSION)
 p=len(fg)/(len(fg)+len(bg));q=balanced_share(p,amount);count=round(n*q)
 # Generate full fixed pools even when quotas differ; seed/order policy is shared by controls.
 f=fg[rng.integers(0,len(fg),n)];b=bg[rng.integers(0,len(bg),n)];order=rng.permutation(n)
 a=np.concatenate([f[:count],b[:n-count]])[order].reshape(side,side,3)
 return a,dict(applied=amount>0,source_share=p,balanced_share=q,reliable_fraction=(len(fg)+len(bg))/len(flat),
               max_relative_contribution=2,version=VERSION)
