"""PROBE ONLY: masked histogram and at-most-512 regional moment samples; not native recovered semantics."""
"""Independent port of recovered feature math.

The recovered histogram's ordered elif branches and anomalous boundaries are
intentional. Histogram and color statistics use different sampling patterns.
No native artifacts or diagnostic probe modules are runtime dependencies.
"""
import numpy as np
from reference_color_app.contracts import FeaturePacket

f = np.float32

def gl_resize(a,w,h):
 ih,iw=a.shape[:2]
 px=(np.arange(w,dtype=np.float32)+np.float32(.5))*np.float32(iw/w)-np.float32(.5)
 py=(np.arange(h,dtype=np.float32)+np.float32(.5))*np.float32(ih/h)-np.float32(.5)
 xx=np.floor(px).astype(int);yy=np.floor(py).astype(int);fx=(px-xx).astype(np.float32);fy=(py-yy).astype(np.float32)
 x0=xx.clip(0,iw-1);x1=(xx+1).clip(0,iw-1);y0=yy.clip(0,ih-1);y1=(yy+1).clip(0,ih-1)
 top=a[y0[:,None],x0[None,:]]*(1-fx[None,:,None])+a[y0[:,None],x1[None,:]]*fx[None,:,None]
 bot=a[y1[:,None],x0[None,:]]*(1-fx[None,:,None])+a[y1[:,None],x1[None,:]]*fx[None,:,None]
 return top*(1-fy[:,None,None])+bot*fy[:,None,None]
def moments(pairs, mode):
 # Mirror column-mean helper's four-at-a-time order, remaining items scalar.
 n=len(pairs);mu=np.zeros(2,np.float32)
 for j in range(2):
  s=np.float32(0);k=0
  while k+3<n:
   s=np.float32(np.float32(np.float32(np.float32(s+pairs[k,j])+pairs[k+1,j])+pairs[k+2,j])+pairs[k+3,j]);k+=4
  while k<n:s=np.float32(s+pairs[k,j]);k+=1
  mu[j]=s/np.float32(n)
 maxabs=float(np.max(np.abs(mu)))
 if mode == 1 and maxabs>.2:mu=mu*np.float32(np.float32(.2)/np.float32(maxabs))
 centered=pairs-mu;cov=np.zeros((2,2),np.float32)
 for i in range(2):
  for j in range(2):
   s=np.float32(0)
   for p in centered:s=np.float32(s+np.float32(p[i]*p[j]))
   cov[i,j]=s/np.float32(n-1 if n!=1 else 1)
 return mu,cov
def histogram(q,mode):
 y=q[:,:,0].ravel()[::20];bins=np.bincount(y,minlength=256);n=len(y)
 trim=int(float(f(n)*f(.2))+.5);d=int((n+2*trim)/3+.5)
 t=[mode^1,-1,-1,-1,-1,255];cum=0
 for i in range(t[0],256):
  cum+=int(bins[i])
  if 2*d-trim<cum and t[4]==-1:t[4]=i
  elif (d-trim)*2<cum and t[3]==-1:t[3]=i
  elif d<cum and t[2]==-1:t[2]=i
  elif d-trim<cum and t[1]==-1:t[1]=i
 valid=all(v>=1 for v in t[1:5])
 if not valid:t=[0,0,255,0,255,255]
 groups=[range(t[0],max(60,t[2]) if valid else 255),range(t[1],t[4] if valid else 256),range(min(170,t[4]) if valid else 0,255)]
 sums=[];counts=[]
 for inds in groups:
  inds=list(inds);counts.append(sum(int(bins[k]) for k in inds));sums.append(sum(int(bins[k])*k for k in inds))
 if all(counts):means=np.array([f(s)/f(c) for s,c in zip(sums,counts)],np.float32)
 else:means=np.full(3,f(sum(sums))/f(sum(counts)) if sum(counts)>0 else f(0),np.float32)
 lo=int(np.searchsorted(np.cumsum(bins),f(n)*f(.01)))
 hi=255-int(np.searchsorted(np.cumsum(bins[::-1]),f(n)*f(.01)))
 if mode==1:
  if valid:t[0]=min(t[1]-1,30);t[5]=max(t[4]+1,225);means[2]=f(means[2]+f(10))
  lo=0;hi=235 if hi<220 else 255
 elif valid:
  if t[2]-t[1]<50:
   if t[2]<51:t[2]=min(t[1]+50,t[3]-1)
   else:t[1]=min(t[1],t[2]-50)
  if t[4]-t[3]<50:
   if 255-t[3]>50:t[4]=t[3]+50
   else:t[3]=max(t[4]-50,t[2]+1)
 return np.array(t),means,[lo,hi],valid
def region_masks(y, t):
 """Native ordered routing, including reachable nonmonotonic bounds.

 Labels A/B/C/D follow region-order-audit.md. Overlap routes deliberately
 bypass t0/t5; independent interval masks do not preserve these branches.
 """
 y = np.asarray(y)
 low_side = y < t[1]
 low_middle = ~low_side & (y < t[2])
 # Low-side ordered jumps: A takes precedence over B, then C, then D.
 at_a = (low_side & (y >= t[2])) | (~low_side & ~low_middle)
 middle_only = at_a & (y < t[3])
 at_b = (at_a & ~middle_only) | (low_side & (y < t[2]) & (y >= t[3]))
 middle_high = at_b & (y < t[4])
 at_c = (at_b & ~middle_high) | (low_side & (y < t[2]) & (y < t[3]) & (y >= t[4]))
 high_only = at_c & (y <= t[5])
 at_d = (low_side & (y < t[2]) & (y < t[3]) & (y < t[4])) | (at_c & ~high_only)
 low_only = at_d & (y >= t[0]) & (y < t[1])
 return [low_middle | low_only, low_middle | middle_only | middle_high, middle_high | high_only]
def saturation(sample,t):
 s=sample.astype(np.float32)
 yy=np.where(s[:,0]<128,s[:,0]*f(.5)*f(1/128),((s[:,0]-128)*f(.5))/127+f(.5))
 u=(s[:,1]-128)*f(.5);u=np.where(s[:,1]<128,u*f(1/128),u/f(127))
 v=(s[:,2]-128)*f(.5);v=np.where(s[:,2]<128,v*f(1/128),v/f(127))
 rgb=np.clip(np.stack([yy+v*f(1.14),yy-u*f(.395)-v*f(.581),yy+u*f(2.032)],axis=1),0,1)
 mx=rgb.max(axis=1);mn=rgb.min(axis=1);delta=mx-mn
 den=np.where((mn+mx)*f(.5)>=f(.5),(f(2)-mx)-mn,mn+mx)
 sat=np.divide(delta,den,out=np.zeros_like(delta),where=(delta>np.finfo(np.float32).eps)&(den!=0))
 groups=region_masks(sample[:,0],t);sums=[];counts=[]
 for mask in groups:
  acc=f(0)
  for val in sat[mask]:acc=f(acc+val)
  sums.append(acc);counts.append(int(mask.sum()))
 if all(counts):return np.array([f(s/f(n))*f(255) for s,n in zip(sums,counts)],np.float32)
 total=f(f(sums[0]+sums[1])+sums[2]);count=sum(counts)
 return np.full(3,f(total/f(count))*f(255) if count else f(0),np.float32)
def extract_features(rgb_u8, mode, spatial_mask):
 """Compute a 63/57-item packet from RGB bytes, without native cache access.

 Center-aligned float32 bilinear resize and UNORM rounding are explicit
 preprocessing hypotheses, not a claim of native byte compatibility.
 """
 if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)) or mode not in (0, 1):
  raise ValueError('mode must be integer 0 or 1 (not bool)')
 if not isinstance(rgb_u8, np.ndarray) or rgb_u8.dtype != np.uint8:
  raise TypeError('rgb_u8 must be a uint8 numpy array')
 if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3 or min(rgb_u8.shape[:2]) == 0:
  raise ValueError('rgb_u8 must have nonempty shape (H, W, 3)')
 mode = int(mode)
 rgb = rgb_u8.astype(np.float32) / f(255)
 ih,iw=rgb.shape[:2];cap=680
 w,h=(cap,int(f(cap)*f(ih)/f(iw))) if iw>=ih and iw>cap else (int(f(cap)*f(iw)/f(ih)),cap) if ih>cap else (iw,ih)
 if min(w,h) < 1:
  raise ValueError('680-pixel resize produces zero short dimension for this aspect ratio')
 a=gl_resize(rgb,w,h)
 yy=f(.299)*a[:,:,0]+f(.587)*a[:,:,1]+f(.114)*a[:,:,2]
 converted=np.stack([yy,(a[:,:,2]-yy)*f(.492)+f(.5),(a[:,:,0]-yy)*f(.877)+f(.5)],axis=-1)
 q=np.rint(np.clip(converted,0,1)*f(255)).astype(np.uint8)
 selected=cv2.resize(spatial_mask.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)>.8
 if selected.sum()<64:raise ValueError('Insufficient reliable region pixels')
 t,luma,endpoints,valid=histogram(q[selected].reshape(-1,1,3),mode)
 dense=q[selected];step=max(1,int(np.ceil(len(dense)/512)));sample=dense[::step];ys=sample[:,0];raw=sample[:,1:].astype(np.float32)/f(255);pairs=raw-f(.5)
 eligible=(ys!=0)&((ys!=255)|(np.abs(pairs).max(axis=1)>=1e-5))
 if mode==1:
  delta=raw-np.array([f(.4431372582912445),f(.610196053981781)])
  mask=np.exp(-f(100)*np.sum(delta*delta,axis=1))
  eligible &= (mask<=.33)|((ys>=25.5)&(ys<=229.5))
 affine=[];covs=[];counts=[]
 for region in region_masks(ys,t):
  selected=eligible&region;points=pairs[selected] if selected.any() else pairs
  mu,cov=moments(points, mode);m=np.eye(3,dtype=np.float32);m[:2,2]=mu
  affine.append(m);covs.append(cov);counts.append(len(points))
 sat=saturation(sample,t)
 if mode==1:
  tail=f(np.clip(f(max(f(affine[2][1,2]*f(2.1)),f(0))+f(.4)),f(.4),f(.8)))
  packet=np.concatenate([np.array(covs).ravel(),np.array(affine).ravel(),sat,t.astype(np.float32),np.zeros(3,np.float32),np.array(endpoints,np.float32),[tail],luma]).astype('<f4')
 else:
  inverses=[]
  for c in covs:
   det=f(f(c[0,0]*c[1,1])-f(c[1,0]*c[0,1]))
   if float(det)<=1e-5:inv=np.eye(2,dtype=np.float32)
   else:inv=np.array([[c[1,1],-c[0,1]],[-c[1,0],c[0,0]]],np.float32)*f(f(1)/det)
   inverses.append(inv)
  packet=np.concatenate([np.array(covs).ravel(),np.array(affine).ravel(),np.array(inverses).ravel(),sat,t.astype(np.float32),luma]).astype('<f4')
 return FeaturePacket(mode, packet, {'preprocessing': 'float32 center-aligned bilinear; round-to-even UNORM; host preprocessing unverified', 'native_byte_compatible': False, 'size':[w,h],'histogram_valid':bool(valid),'thresholds':t.tolist(),'region_sample_counts':counts,'luma_means':luma.tolist(),'saturation':sat.tolist()})
