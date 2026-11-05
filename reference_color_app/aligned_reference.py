"""Research: certify spatial correspondence before fitting a color transform.

Own algorithm, not native reconstruction. Separate spatial tile holdout.
"""
import numpy as np
from PIL import Image
from reference_color_app.contracts import Mapping

def fit_aligned(source,reference):
 for a in (source,reference):
  if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<16:
   return dict(accepted=False,reason='Unsupported image dimensions or precision')
 h,w=source.shape[:2];rh,rw=reference.shape[:2]
 if abs((w/h)/(rw/rh)-1)>.01:return dict(accepted=False,reason='Different aspect ratios')
 size=(max(16,round(w*min(1,256/max(h,w)))),max(16,round(h*min(1,256/max(h,w)))))
 a=np.asarray(Image.fromarray(source).resize(size,Image.Resampling.BILINEAR)).astype(float)/255
 b=np.asarray(Image.fromarray(reference).resize(size,Image.Resampling.BILINEAR)).astype(float)/255
 ya=a@np.array([.299,.587,.114]);yb=b@np.array([.299,.587,.114])
 ga=np.concatenate([np.diff(ya,axis=0).ravel(),np.diff(ya,axis=1).ravel()])
 gb=np.concatenate([np.diff(yb,axis=0).ravel(),np.diff(yb,axis=1).ravel()])
 if min(ga.std(),gb.std())<.005:return dict(accepted=False,reason='Insufficient spatial structure')
 correlation=float(np.corrcoef(ga,gb)[0,1])
 if correlation<.95:return dict(accepted=False,reason='Spatial structure differs',gradient_correlation=correlation)
 yy,xx=np.indices(ya.shape);train=((xx//8+yy//8)%2==0).ravel()
 x=np.column_stack([a.reshape(-1,3),np.ones(a.shape[0]*a.shape[1])]);y=b.reshape(-1,3)
 if np.linalg.cond(x[train])>200:return dict(accepted=False,reason='Ill-conditioned color samples')
 coefficients=[]
 for c in range(3):
  usable=train&(y[:,c]>2/255)&(y[:,c]<253/255)
  if usable.sum()<128:return dict(accepted=False,reason='Insufficient unclipped training pixels')
  coefficients.append(np.linalg.lstsq(x[usable],y[usable,c],rcond=None)[0])
 matrix=np.stack(coefficients,axis=1)
 error=np.abs(np.clip(x[~train]@matrix,0,1)-y[~train])*255
 all_error=(np.abs(np.clip(x@matrix,0,1)-y)*255).mean(axis=1).reshape(ya.shape)
 tile_errors=[float(all_error[j:j+8,i:i+8].mean()) for j in range(0,ya.shape[0],8) for i in range(0,ya.shape[1],8)]
 diagnostics=dict(accepted=False,gradient_correlation=correlation,holdout_mae=float(error.mean()),
  holdout_p95=float(np.quantile(error,.95)),holdout_pixels=int((~train).sum()),max_tile_mae=max(tile_errors),native_policy=False)
 if np.max(np.abs(matrix[3]))>.3 or np.max(np.sum(np.abs(matrix[:3]),axis=0))>3:
  return dict(diagnostics,reason='Color transform exceeds supported bounds')
 if diagnostics['holdout_mae']>1.5 or diagnostics['holdout_p95']>4:
  return dict(diagnostics,reason='Independent spatial holdout failed')
 if diagnostics['max_tile_mae']>6:return dict(diagnostics,reason='Local content or color mismatch')
 return dict(diagnostics,accepted=True,matrix=matrix.tolist(),version='aligned-affine-trial-v1')

def make_mapping(fit):
 if not fit.get('accepted'):raise ValueError('Uncertified transform')
 axis=np.linspace(0,1,64,dtype=np.float32);rgb=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
 x=np.concatenate([rgb,np.ones((*rgb.shape[:3],1),np.float32)],axis=-1)
 transformed=np.clip(x@np.asarray(fit['matrix']),0,1).astype(np.float32)
 return Mapping(np.stack([.2*rgb+.8*transformed,transformed]),dict(aligned_reference=fit))
