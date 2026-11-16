"""Frozen development interventions and measurements; no product inference masks."""
import numpy as np
from PIL import Image, ImageFilter

LUMA = np.array([.299, .587, .114])
GREEN = np.array([-24., 24*.413/.587, -24.])


def validate(rgb, mask):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty RGB8')
    if not isinstance(mask, np.ndarray) or mask.shape != rgb.shape[:2] or mask.dtype != bool:
        raise ValueError('Expected same-size boolean mask')


def edit_weight(mask, coverage):
    protected = np.array(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(11))) > 0
    editable = ~protected
    if not editable.any(): return np.zeros(mask.shape, float), editable
    # Eight 3x3 expansions measure capped chessboard distance without a new dependency.
    edge = np.zeros(mask.shape, float)
    expanded = Image.fromarray(protected.astype(np.uint8)*255)
    for _ in range(8):
        edge += np.asarray(expanded) == 0
        expanded = expanded.filter(ImageFilter.MaxFilter(3))
    edge /= 8.
    if coverage == 1: return edge, editable
    x = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    cutoff = float(np.quantile(x[editable], coverage, method='linear'))
    ramp = np.clip((cutoff+4.-x)/8., 0., 1.)
    return edge*ramp, editable


def make_variants(rgb, mask):
    validate(rgb, mask); original=rgb.astype(float); rows=[]

    def add(name, raw, weight=None, editable=None):
        if not np.isfinite(raw).all(): raise ValueError('Nonfinite intervention')
        pixels=np.rint(raw.clip(0,255)).astype(np.uint8)
        delta=pixels.astype(float)-original
        info=dict(person_max_change=int(np.abs(delta[mask]).max()) if mask.any() else 0,
                  luminance_mean_change=float((delta@LUMA).mean()),
                  preclip_channel_fraction=float(((raw<0)|(raw>255)).mean()),
                  endpoint_channel_fraction=float(((pixels==0)|(pixels==255)).mean()))
        if weight is not None:
            if not np.array_equal(pixels[mask],rgb[mask]): raise ValueError('Person was edited')
            info.update(editable_pixels=int(editable.sum()),
                        mean_edit_weight=float(weight[editable].mean()) if editable.any() else 0.,
                        edited_background_fraction=float((weight[editable]>0).mean()) if editable.any() else 0.)
        rows.append((name,pixels,info))

    add('identity',original);add('identity_repeat',original)
    for coverage in (.25,.5,1.):
        weight,editable=edit_weight(mask,coverage)
        for label,sign in [('green',1),('magenta',-1)]:
            add(f'background_{label}_{round(coverage*100)}', original+weight[...,None]*GREEN*sign,weight,editable)
    add('whole_green',original+GREEN);add('whole_magenta',original-GREEN)
    add('whole_contrast',(original-127.5)*1.1+127.5)
    add('whole_desaturate',original*.8+(original@LUMA)[...,None]*.2)
    return rows


def target_regions(rgb, mask):
    validate(rgb,mask); x=rgb.astype(float); y=x@LUMA
    return dict(person=mask.copy(),background=~mask,
                neutral_proxy=(x.max(-1)-x.min(-1)<=20)&(y>=40)&(y<=220),
                whole=np.ones(mask.shape,bool))


def region_delta(actual, baseline, mask):
    validate(actual,mask);validate(baseline,mask)
    if not mask.any():return None
    a,b=actual[mask].astype(float),baseline[mask].astype(float);delta=a-b
    mae=float(np.abs(delta).mean())
    return dict(pixels=int(mask.sum()),rgb_mae=mae,rgb_mean_delta=delta.mean(0).tolist(),
                luminance_mean_delta=float((delta@LUMA).mean()),
                saturation_range_mean_delta=float(((a.max(-1)-a.min(-1))-(b.max(-1)-b.min(-1))).mean()),
                sensitive_over_one_code=bool(mae>1.))
