"""Research v2: dense weighted statistics, no random atlas or sparse sampling.

Reuses recovered packet layout / histogram rules, changes statistical estimator.
This is not a byte-compatible port of native feature extraction.
"""
from pathlib import Path
import sys,inspect
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skin-protection-20261019'))
from reference_color_app import features as base
from reference_color_app.contracts import FeaturePacket
from balance import balanced_share

# Keep the audited ordered histogram rules, replacing only its count acquisition.
_source=inspect.getsource(base.histogram)
_source=_source.replace('def histogram(q,mode):','def from_bins(bins,mode):')
_old='y=q[:,:,0].ravel()[::20];bins=np.bincount(y,minlength=256);n=len(y)'
if _old not in _source:raise RuntimeError('Feature dependency changed; review histogram adapter')
_source=_source.replace(_old,'n=int(bins.sum())')
_scope={'np':np,'f':np.float32};exec(compile(_source,'audited_histogram_adapter','exec'),_scope)
_histogram=_scope['from_bins']
VERSION='dense-weighted-packet-v3-stable-moments'

def extract_dense(rgb,mode,mask=None,amount=0,*,goal_share=.5):
 balanced_share(.5,amount,goal=goal_share)
 if isinstance(mode,(bool,np.bool_)) or mode not in (0,1):raise ValueError('mode must be 0 or 1')
 if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:raise ValueError('Expected RGB8')
 h,w=rgb.shape[:2];weights=np.ones(h*w,np.float64);applied=False;diagnostics={}
 if mask is not None:
  m=np.asarray(mask)
  if m.shape!=(h,w) or not np.isfinite(m).all() or m.min()<0 or m.max()>1:raise ValueError('Invalid mask')
  fg=m.ravel()>.8;bg=m.ravel()<.2;nf=int(fg.sum());nb=int(bg.sum())
  if min(nf,nb)>=max(16,h*w*.01) and nf+nb>=.5*h*w:
   p=nf/(nf+nb);q=balanced_share(p,amount,goal=goal_share)
   # Uncertain pixels keep unit weight; zero strength is the full dense control.
   weights[fg]=q/p;weights[bg]=(1-q)/(1-p)
   applied=q!=p;diagnostics=dict(source_share=p,balanced_share=q)
 weights/=weights.sum()
 a=rgb.reshape(-1,3).astype(np.float64)/255
 y=a@np.array([.299,.587,.114]);uv=np.stack([(a[:,2]-y)*.492+.5,(a[:,0]-y)*.877+.5],axis=1)
 qrgb=np.rint(np.clip(np.column_stack([y,uv]),0,1)*255).astype(np.uint8)
 ys=qrgb[:,0];raw=qrgb[:,1:].astype(np.float64)/255;pairs=raw-.5
 # Fixed normalized histogram mass avoids dependence on image duplication size.
 hist=np.bincount(ys,weights=weights,minlength=256)
 bins=np.rint(hist*10_000_000).astype(np.int64);bins[np.argmax(bins)]+=10_000_000-int(bins.sum())
 t,luma,endpoints,valid=_histogram(bins,mode)
 eligible=(ys!=0)&((ys!=255)|(np.abs(pairs).max(axis=1)>=1e-5))
 if mode==1:
  delta=raw-np.array([.4431372582912445,.610196053981781]);skin_like=np.exp(-100*np.sum(delta*delta,axis=1))
  eligible&=(skin_like<=.33)|((ys>=25.5)&(ys<=229.5))
 affine=[];covs=[]
 for region in base.region_masks(ys,t):
  selected=eligible&region
  if not selected.any():selected=np.ones(len(ys),bool)
  v=pairs[selected];ww=weights[selected];ww=ww/ww.sum()
  # Work relative to an observed color so constant samples have EXACT zero
  # variance. Direct weighted summation invents ~1e-28 covariance, which can
  # underflow inside the recovered float32 matrix solver.
  relative=v-v[0];shift=(relative*ww[:,None]).sum(axis=0);mu=v[0]+shift
  maxabs=np.abs(mu).max()
  if mode==1 and maxabs>.2:
   mu*=.2/maxabs;centered=v-mu
  else:centered=relative-shift
  cov=(centered*ww[:,None]).T@centered
  mat=np.eye(3);mat[:2,2]=mu;affine.append(mat);covs.append(cov)
 # Weighted saturation estimator using the same recovered YUV reconstruction.
 s=qrgb.astype(np.float64)
 yy=np.where(s[:,0]<128,s[:,0]*.5/128,(s[:,0]-128)*.5/127+.5)
 u=np.where(s[:,1]<128,(s[:,1]-128)*.5/128,(s[:,1]-128)*.5/127)
 v=np.where(s[:,2]<128,(s[:,2]-128)*.5/128,(s[:,2]-128)*.5/127)
 reconstructed=np.clip(np.stack([yy+v*1.14,yy-u*.395-v*.581,yy+u*2.032],axis=1),0,1)
 mx=reconstructed.max(axis=1);mn=reconstructed.min(axis=1);delta=mx-mn
 den=np.where((mn+mx)*.5>=.5,2-mx-mn,mn+mx)
 sat=np.divide(delta,den,out=np.zeros_like(delta),where=(delta>np.finfo(np.float32).eps)&(den!=0))
 groups=base.region_masks(ys,t);counts=[weights[g].sum() for g in groups];sums=[(sat[g]*weights[g]).sum() for g in groups]
 saturation=np.array([s/c for s,c in zip(sums,counts)])*255 if min(counts)>0 else np.full(3,sum(sums)/sum(counts)*255 if sum(counts)>0 else 0)
 if mode==1:
  tail=np.clip(max(affine[2][1,2]*2.1,0)+.4,.4,.8)
  packet=np.concatenate([np.asarray(covs).ravel(),np.asarray(affine).ravel(),saturation,t,np.zeros(3),endpoints,[tail],luma])
 else:
  inverse=[np.eye(2) if np.linalg.det(c)<=1e-5 else np.linalg.inv(c) for c in covs]
  packet=np.concatenate([np.asarray(covs).ravel(),np.asarray(affine).ravel(),np.asarray(inverse).ravel(),saturation,t,luma])
 return FeaturePacket(mode,packet.astype('<f4'),dict(version=VERSION,applied=applied,native_byte_compatible=False,
   pixels=h*w,estimator='dense float64 weighted population moments; normalized integer histogram mass',**diagnostics))
