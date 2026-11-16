"""Model-Y-preserving symmetric in-gamut controls, separate from frozen v1."""
import numpy as np
from background_intervention import validate, edit_weight, target_regions, region_delta

MODEL_YCC=np.array([[.2126,.7152,.0722],[-.1146,-.3854,.5],[.5,-.4542,-.0458]])
VECTOR=np.array([-24.,24*(.2126+.0722)/.7152,-24.])


def chroma_shift(rgb,weight,sign):
    validate(rgb,np.zeros(rgb.shape[:2],bool))
    if not isinstance(weight,np.ndarray) or weight.shape!=rgb.shape[:2] or not np.isfinite(weight).all() or (weight<0).any() or (weight>1).any() or sign not in (-1,1):
        raise ValueError('Invalid intervention weights/direction')
    original=rgb.astype(float)
    scale=np.minimum(1.,(np.minimum(original,255-original)/np.abs(VECTOR)).min(-1))
    raw=original+sign*VECTOR*scale[...,None]*weight[...,None]
    delta=raw-original
    error=float(np.abs(delta@MODEL_YCC[0]).max())
    if error>1e-10 or raw.min()<0 or raw.max()>255:raise ValueError('Float model-Y/gamut control failed')
    return raw,dict(symmetric_scale_mean=float(scale.mean()),symmetric_scale_zero_fraction=float((scale==0).mean()),
                    float_model_y_max_abs_delta=error)


def make_variants(rgb,mask):
    validate(rgb,mask);x=rgb.astype(float);rows=[]

    def add(name,raw,weight=None,editable=None,chroma=None):
        if not np.isfinite(raw).all():raise ValueError('Nonfinite intervention')
        pixels=np.rint(raw.clip(0,255)).astype(np.uint8);delta=pixels.astype(float)-x;ycc=delta@MODEL_YCC.T
        info=dict(person_max_change=int(np.abs(delta[mask]).max()) if mask.any() else 0,
                  luminance_mean_change=float((delta@np.array([.299,.587,.114])).mean()),
                  preclip_channel_fraction=float(((raw<0)|(raw>255)).mean()),
                  endpoint_channel_fraction=float(((pixels==0)|(pixels==255)).mean()),
                  model_ycc_mean_delta=ycc.reshape(-1,3).mean(0).tolist(),
                  model_y_max_abs_quantized_delta=float(np.abs(ycc[...,0]).max()),reference_rgb_mae=float(np.abs(delta).mean()))
        if chroma is not None:
            if info['model_y_max_abs_quantized_delta']>.500001:raise ValueError('Quantized model-Y control failed')
            info.update(chroma)
        if weight is not None:
            if not np.array_equal(pixels[mask],rgb[mask]):raise ValueError('Person changed')
            info.update(editable_pixels=int(editable.sum()),mean_edit_weight=float(weight[editable].mean()) if editable.any() else 0.,
                        edited_background_fraction=float((weight[editable]>0).mean()) if editable.any() else 0.)
        rows.append((name,pixels,info))

    add('identity',x);add('identity_repeat',x)
    for coverage in (.25,.5,1.):
        w,e=edit_weight(mask,coverage)
        for label,sign in [('green',1),('magenta',-1)]:
            raw,stats=chroma_shift(rgb,w,sign);add(f'background_{label}_{round(coverage*100)}',raw,w,e,stats)
    for label,sign in [('green',1),('magenta',-1)]:
        raw,stats=chroma_shift(rgb,np.ones(mask.shape),sign);add('whole_'+label,raw,chroma=stats)
    add('whole_contrast',(x-127.5)*1.1+127.5)
    add('whole_desaturate',x*.8+(x@MODEL_YCC[0])[...,None]*.2)
    return rows
