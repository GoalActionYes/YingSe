"""Research-only eligibility: visible confident face, not a person detector.

Uses the existing segmenter. Unsupported/ambiguous scenes return an empty mask
so composition adaptation falls back to the unweighted dense control.
"""
from pathlib import Path
import sys,math
import numpy as np
import cv2

VERSION='visible-face-eligibility-v4'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skin-protection-20261019'))


def face_evidence(category,face_confidence):
 c=np.asarray(category);f=np.asarray(face_confidence)
 if (c.ndim!=2 or min(c.shape,default=0)<1 or f.shape!=c.shape or
     not np.issubdtype(c.dtype,np.integer) or not np.isfinite(f).all() or
     f.min()<0 or f.max()>1 or c.min()<0 or c.max()>5):
  raise ValueError('Invalid segmentation evidence')
 selected=((c==3)&(f>=.85)).astype(np.uint8)
 _,_,stats,_=cv2.connectedComponentsWithStats(selected,connectivity=8)
 largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
 minimum=max(16,math.ceil(c.size*.0001))
 return dict(eligible=largest>=minimum,largest_face_component=largest,
   minimum_face_component=minimum,face_confidence_threshold=.85,
   high_confidence_face_pixels=int(selected.sum()))


def segment(rgb):
 if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or
     rgb.shape[2]!=3 or min(rgb.shape[:2])<1):raise ValueError('Expected RGB8')
 from reference_color_app import skin
 h,w=rgb.shape[:2];ratio=min(1,512/max(h,w))
 small=cv2.resize(rgb,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
 with skin._engine_lock:
  engine=skin._get_engine();mp=skin._mp
  result=engine.segment(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(small)))
  cat=result.category_mask.numpy_view().copy().squeeze()
  confidence=[x.numpy_view().copy().squeeze() for x in result.confidence_masks]
 evidence=face_evidence(cat,confidence[3])
 conf=sum(confidence[k] for k in (1,2,3,4))
 ramp=np.clip((conf-.45)/.4,0,1)
 mask=(ramp*ramp*(3-2*ramp)*np.isin(cat,(1,2,3,4))).astype(np.float32)
 raw_detected=int(np.count_nonzero(mask>.5))>=max(64,int(mask.size*.01))
 if not raw_detected:mask.fill(0)
 mask=cv2.GaussianBlur(mask,(0,0),1.2)
 raw_fraction=float((mask>.5).mean())
 allowed=bool(raw_detected and evidence['eligible'])
 if not allowed:mask.fill(0)
 diag=dict(version=VERSION,person_detected=allowed,person_fraction=float((mask>.5).mean()),
           raw_person_fraction=raw_fraction,face_evidence=evidence,
           reason='Visible face evidence present' if allowed else 'No sufficient visible face evidence; unweighted fallback',
           model_sha256=skin.MODEL_SHA256)
 return mask,cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),diag


def fullmask(mask,rgb):
 return np.clip(cv2.resize(mask,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR),0,1)
