"""Known synthetic styles for evaluation only. Human masks are not model inputs."""
import numpy as np

PROFILES=('identity','global_contrast','global_desaturate','region_color_luminance','region_tone_saturation')


def stylize(rgb,person,profile):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB8')
    mask=np.asarray(person,dtype=float)
    if mask.shape!=rgb.shape[:2] or not np.isfinite(mask).all() or np.any(mask<0) or np.any(mask>1):raise ValueError('Invalid oracle mask')
    x=rgb.astype(float)/255
    gray=(x @ [.299,.587,.114])[...,None]
    if profile=='identity':return rgb.copy()
    if profile=='global_contrast':result=(x-.5)*1.12+.5
    elif profile=='global_desaturate':result=gray+(x-gray)*.6
    elif profile=='region_color_luminance':
        foreground=x*[.92,.99,1.04]+[0,.015,.02]
        background=x*[1.04,1,.94]+[.025,0,0]
        result=mask[...,None]*foreground+(1-mask[...,None])*background
    elif profile=='region_tone_saturation':
        foreground=gray+(x-gray)*.65+.03
        background=(gray+(x-gray)*1.12-.5)*1.12+.5-.025
        result=mask[...,None]*foreground+(1-mask[...,None])*background
    else:raise ValueError('Unknown controlled style')
    return np.rint(np.clip(result,0,1)*255).astype(np.uint8)
