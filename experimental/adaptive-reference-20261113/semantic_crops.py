"""Learned regional-crop candidate. Only RGB and automatic probabilities enter it."""
import time
import numpy as np
from PIL import Image,ImageFilter
from canonical_model import validate_rgb


def valid_probability(rgb,probability):
    validate_rgb(rgb);p=np.asarray(probability,dtype=float)
    if p.shape!=rgb.shape[:2] or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid automatic person probability')
    return p


def textured(rgb,box):
    left,top,right,bottom=box
    if min(right-left,bottom-top)<48:return False
    return float((rgb[top:bottom,left:right].astype(float) @ [.299,.587,.114]).std())>=8


def choose_crops(rgb,probability):
    p=valid_probability(rgb,probability);h,w=p.shape
    yy,xx=np.where(p>.6)
    if len(xx)<h*w*.03:return dict(accepted=False,reason='Insufficient reliable person support')
    foreground=(max(0,int(xx.min()-w*.05)),max(0,int(yy.min()-h*.05)),
                min(w,int(xx.max()+1+w*.05)),min(h,int(yy.max()+1+h*.05)))
    if not textured(rgb,foreground):return dict(accepted=False,reason='Uninformative person crop')
    background=None
    for scale in (.6,.4):
        hh,ww=round(h*scale),round(w*scale);choices=[]
        for y in (0,(h-hh)//2,h-hh):
            for x in (0,(w-ww)//2,w-ww):
                box=(x,y,x+ww,y+hh);purity=float(p[y:y+hh,x:x+ww].mean())
                if purity<.1 and textured(rgb,box):choices.append((purity,box))
        if choices:
            background=min(choices,key=lambda item:item[0])[1];break
    if background is None:return dict(accepted=False,reason='No reliable textured background crop')
    return dict(accepted=True,foreground=list(foreground),background=list(background))


def blend_residual(global_rgb,foreground_rgb,background_rgb,person):
    p=valid_probability(global_rgb,person)
    for value in (foreground_rgb,background_rgb):
        validate_rgb(value)
        if value.shape!=global_rgb.shape:raise ValueError('Regional output dimensions differ')
    p=np.asarray(Image.fromarray(np.rint(p*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)),dtype=float)/255
    confidence=np.clip((np.abs(p-.5)-.1)/.4,0,1)
    if confidence.sum()<1:return global_rgb.copy()
    base=global_rgb.astype(float)
    delta=p[...,None]*(foreground_rgb.astype(float)-base)+(1-p[...,None])*(background_rgb.astype(float)-base)
    mean=(delta*confidence[...,None]).sum(axis=(0,1))/confidence.sum()
    result=base+.5*confidence[...,None]*np.clip(delta-mean,-24,24)
    return np.rint(np.clip(result,0,255)).astype(np.uint8)


class SemanticCropCandidate:
    def __init__(self,model):self.model=model

    def predict(self,target,reference,target_probability,reference_probability):
        start=time.perf_counter()
        global_rgb,_=self.model.predict(target,reference)
        source=choose_crops(target,target_probability);ref=choose_crops(reference,reference_probability)
        diagnostic=dict(target_crops=source,reference_crops=ref,applied=False,
                        scope='Learned crop residual; not independently validated quality')
        if not source['accepted'] or not ref['accepted']:
            return global_rgb,global_rgb,dict(diagnostic,seconds=time.perf_counter()-start)
        from gpu_mapping import PreparedMapping
        regional=[]
        for role in ('foreground','background'):
            x0,y0,x1,y1=source[role];a=target[y0:y1,x0:x1]
            x0,y0,x1,y1=ref[role];b=reference[y0:y1,x0:x1]
            self.model.predict(a,b)
            regional.append(PreparedMapping(*self.model.last_luts,device=self.model.device).apply(target))
        result=blend_residual(global_rgb,*regional,target_probability)
        return result,global_rgb,dict(diagnostic,applied=True,seconds=time.perf_counter()-start)
