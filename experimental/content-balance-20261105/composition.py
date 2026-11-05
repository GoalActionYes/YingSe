"""v3 research: adapt reference contribution toward target composition only.

One global color map, no region-wise rendering. Dense extraction is still an
independent estimator, not proof of Jianying's exact input extraction logic.
"""
import numpy as np
from balance import balanced_share
from dense_features import extract_dense

VERSION='reference-composition-v3'


def reliable_share(rgb,mask):
 if mask is None:return None
 m=np.asarray(mask)
 if m.shape!=rgb.shape[:2] or not np.isfinite(m).all() or m.min()<0 or m.max()>1:
  raise ValueError('Invalid mask')
 nf=int((m>.8).sum());nb=int((m<.2).sum());n=m.size
 if min(nf,nb)<max(16,n*.01) or nf+nb<n*.5:return None
 return nf/(nf+nb)


def extract_pair(target,reference,target_mask,reference_mask,amount=.5):
 balanced_share(.5,amount)
 # Validate image contracts before touching masks. Source is ALWAYS unweighted.
 source=extract_dense(target,0)
 reference_plain=extract_dense(reference,1)
 p=reliable_share(target,target_mask);r=reliable_share(reference,reference_mask)
 diagnostics=dict(version=VERSION,target_share=p,reference_share=r,amount=amount,
                  applied=False,native_byte_compatible=False)
 if amount==0 or p is None or r is None or p==r:
  diagnostics['reason']='Disabled, unavailable reliable regions, or equal composition'
  return source,reference_plain,diagnostics
 packet=extract_dense(reference,1,reference_mask,amount,goal_share=p)
 diagnostics.update(applied=packet.diagnostics['applied'],
                    adjusted_reference_share=packet.diagnostics['balanced_share'])
 return source,packet,diagnostics
