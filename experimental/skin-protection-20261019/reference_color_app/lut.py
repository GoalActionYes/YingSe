"""Portable dual LUT reconstruction; no native or GPU runtime dependencies.

RGBA8 round-to-even is a documented hypothesis. System GL frozen readbacks
agree within one byte for two cases; this does not establish native equivalence.
"""
import numpy as np
from .contracts import FeaturePacket, Mapping
from .parameters import solve_parameters, _tables
F = np.float32


def _identity_cube():
    # Native atlas tile columns R, rows G, tile index B. Truncate float32.
    axis = (np.arange(64, dtype=np.float32)/F(63)*F(255)).astype(np.uint8)
    return np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),axis=-1).astype(np.float32)/F(255)


def _luma_weights(y, bounds):
    """GLSL if/else-if priority is significant for unsorted thresholds."""
    y=np.asarray(y,np.float32)
    w=np.zeros((*y.shape,3),np.float32)
    remaining=np.ones(y.shape,bool)
    m=remaining & (y>=bounds[0]) & (y<bounds[1]);remaining &= ~m
    w[...,0][m]=(bounds[1]-y[m])/(bounds[1]-bounds[0]);w[...,1][m]=F(1)-w[...,0][m]
    m=remaining & (y>=bounds[1]) & (y<bounds[2]);remaining &= ~m
    w[...,1][m]=F(1)
    m=remaining & (y>=bounds[2]) & (y<bounds[3]);remaining &= ~m
    w[...,1][m]=(bounds[3]-y[m])/(bounds[3]-bounds[2]);w[...,2][m]=F(1)-w[...,1][m]
    m=remaining & (y>=bounds[3]) & (y<=F(1));remaining &= ~m
    w[...,2][m]=F(1)
    w[...,0][remaining & (y<bounds[0]) & (y>=0)]=F(1)
    return w/np.sum(w,axis=-1,keepdims=True)


def _shade(rgb, p, strength, mid_curve=False, without_uv=False):
    """Recovered shader arithmetic, including original float32 expression order."""
    rgb = np.asarray(rgb, np.float32)
    y = F(.299)*rgb[...,0]+F(.587)*rgb[...,1]+F(.114)*rgb[...,2]
    uv = np.stack([(rgb[...,2]-y)*F(.492)+F(.5),
                   (rgb[...,0]-y)*F(.877)+F(.5)], axis=-1)
    y=np.clip(y,0,1); uv=np.clip(uv,0,1)
    original_y=y.copy()
    if without_uv:
        mask=F(.210275)
    else:
        delta=uv-np.array([.44314,.6102],np.float32)
        d=np.sqrt(np.sum(delta*delta,axis=-1))
        mask=np.clip(np.exp(-d*d*F(100)),0,1)
    limit=F(1)-np.maximum(F(0),(y-F(.85))/F(.15))
    shadow=np.clip(p[0,0]*(p[0,2]/(F(1)+p[0,3]*mask)+(p[1,0]+mask*limit*p[1,1])),-1,1)
    high=np.clip(p[0,1]*(p[1,2]/(F(1)+p[1,3]*mask*limit)+(p[2,0]+mask*limit*p[2,1])),-1,1)
    tables = _tables()
    curves = np.stack([tables[a] for a in (0x185141be0, 0x185141ce0, 0x185141de0)], axis=-1).astype(np.float32)/F(255)
    idx=np.minimum((y*F(256)).astype(int),255)
    values=curves[idx]
    tmp=np.where(shadow<0,values[...,1 if mid_curve else 0],values[...,2])
    y=(tmp-y)*np.abs(shadow)+y
    tmp=np.where(high<0,y*np.power(F(.71),y),y*np.power(F(1.4),y))
    y=(tmp-y)*np.abs(high)+y
    if p[2,2]>1: y=(y-F(.5))*max(p[2,3],F(1))+F(.5)
    y=np.clip(y*(p[3,1]-p[3,0])+p[3,0],0,1)
    if without_uv:
        return np.stack([y,y,y],axis=-1)
    uv=uv-F(.5)
    def gaussian(mean,c):
        diff=uv-mean
        a=diff[...,0]*c[0]+diff[...,1]*c[1]
        b=diff[...,0]*c[2]+diff[...,1]*c[3]
        return np.exp((diff[...,0]*a+diff[...,1]*b)*F(-.5))
    gs=[gaussian(p[3,2:4],p[4]),
        gaussian(p[5,:2],np.r_[p[5,2:4],p[6,:2]]),
        gaussian(p[6,2:4],p[7])]
    w=_luma_weights(original_y,p[8])
    def affine(start):
        return np.stack([p[start+i,0]*uv[...,0]+p[start+i,1]*uv[...,1]+p[start+i,2] for i in range(3)],axis=-1)
    vec=np.concatenate([uv,np.ones((*y.shape,1),np.float32)],axis=-1)
    a,b,c=(affine(i) for i in (9,12,15))
    shift,shift_mid,shift_shadow=(affine(i) for i in (18,21,24))
    mask=mask[...,None]
    shift=mask*F(.4)*vec+(F(1)-mask*F(.4))*shift
    a=gs[0][...,None]*a+(F(1)-gs[0][...,None])*shift_shadow
    b=gs[1][...,None]*b+(F(1)-gs[1][...,None])*shift_mid
    c=gs[2][...,None]*c+(F(1)-gs[2][...,None])*shift
    shift=F(.8)*shift+F(.2)*vec
    mix=w[...,0,None]*a+w[...,1,None]*b+w[...,2,None]*c
    high_uv=np.zeros_like(vec)
    for row in range(3):
        for group in range(3):
            q=p[27+row*3+group]
            high_uv[...,row]+=q[0]*mix[...,0]*w[...,group]+q[1]*mix[...,1]*w[...,group]+q[2]*mix[...,2]*w[...,group]
    t=F(strength); d=np.clip((t-F(.75))*F(3),0,1)
    mix=(F(1)-d)*mix+d*high_uv
    tmp=w[...,0]*p[36,0]+w[...,1]*p[36,1]+w[...,2]*p[36,2]
    tmp=np.clip(F(1)+(tmp-F(1))*max(t*F(1.25),F(1)),F(.5),F(1.1))
    d=np.sqrt(uv[...,0]*uv[...,0]+uv[...,1]*uv[...,1])
    d=np.where(d>F(.47),F(1),(F(1)-np.cos(d*F(6.594)))/F(2))
    d=np.clip(d,0,F(.5))*np.clip(t+F(1.3),F(.3),F(1))
    mix=d[...,None]*vec+(F(1)-d[...,None])*mix
    mix=mask*shift+((F(1)-mask)*tmp[...,None])*mix
    tmp=np.clip(F(.3)-original_y,0,F(.3))*F(2)
    mix=tmp[...,None]*vec+(F(1)-tmp[...,None])*mix
    uvw=np.clip(t,0,1)*mix+(F(1)-np.clip(t,0,1))*vec
    return np.clip(np.stack([y+uvw[...,1]*F(1.140),y-uvw[...,0]*F(.395)-uvw[...,1]*F(.581),y+uvw[...,0]*F(2.032)],axis=-1),0,1)


def _sample(cube,rgb):
    x=np.clip(np.asarray(rgb,np.float32),0,1)*F(63)
    lo=np.floor(x).astype(np.int32);hi=np.minimum(lo+1,63);f=x-lo.astype(np.float32)
    out=np.zeros_like(x)
    for r in (0,1):
        for g in (0,1):
            for b in (0,1):
                ids=[hi[...,i] if bit else lo[...,i] for i,bit in enumerate((r,g,b))]
                weight=np.prod(np.stack([f[...,i] if bit else 1-f[...,i] for i,bit in enumerate((r,g,b))],axis=-1),axis=-1)
                out+=cube[tuple(ids)]*weight[...,None]
    return out


def _compose(layers,rgb,strength):
    t=F(strength);rgb=np.asarray(rgb,np.float32)
    first=_sample(layers[0],rgb)
    area=bool(t>=F(.80001))
    res=_sample(layers[1],rgb) if area else first
    w=F(np.clip(t/F(.8),F(0),F(1)));w=F(w*w*F(F(3)-F(2)*w))
    if area:w=F(1)
    mw=F(F(F(1)-t)/F(.2))
    if mw>F(1):mw=F(0)
    res=mw*first+F(F(1)-mw)*res
    return rgb*F(F(1)-w)+res*w



def _strength(value):
    if isinstance(value, (bool, str)) or np.ndim(value) != 0:
        raise ValueError("Strength must be a finite scalar in [0,1]")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Strength must be a finite scalar in [0,1]") from exc
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Strength must be a finite scalar in [0,1]")
    return F(value)


def _layers(mapping):
    if not isinstance(mapping, Mapping):
        raise ValueError("Expected Mapping")
    a = np.asarray(mapping.layers)
    if a.shape != (2,64,64,64,3) or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):
        raise ValueError("Expected two RGB-axis 64 cubed layers")
    if not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):
        raise ValueError("Mapping layers must be finite normalized RGB")
    return a.astype(np.float32,copy=False)


def build_mapping(source: FeaturePacket, reference: FeaturePacket) -> Mapping:
    for packet,mode,length in ((source,0,63),(reference,1,57)):
        if not isinstance(packet,FeaturePacket) or packet.mode != mode:
            raise ValueError("Expected source mode0 and reference mode1 feature packets")
        a=np.asarray(packet.values)
        if a.shape != (length,) or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a) or not np.isfinite(a).all():
            raise ValueError("Invalid finite feature packet shape")
    p, diagnostics = solve_parameters(source.values,reference.values)
    if not diagnostics["consumed_values_finite"]:
        raise ValueError("Nonfinite consumed shader parameters")
    identity = _identity_cube()
    layers=[]
    for strength in (.8,1.):
        raw=_shade(identity,p,strength,mid_curve=diagnostics["scalar"]["flag"],without_uv=diagnostics["without_uv"])
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite generated LUT")
        layers.append(np.rint(raw*F(255)).astype(np.uint8).astype(np.float32)/F(255))
    return Mapping(np.stack(layers), {
        "evidence_version":"2026-10-16-render-audit-system-gl-v1",
        "parameters":diagnostics,"quantization":"RGBA8 round-to-even hypothesis",
        "layer_axes":"RGB", "source":source.diagnostics,"reference":reference.diagnostics,
        "limitations":["Not native process GPU capture or bit equivalence", "UI /100 strength mapping remains a hypothesis", "GPU transcendental, UNORM ties, texture interpolation and host preprocessing may differ", "System GL oracle covers one color and one gray feature case"],
    })


def apply_mapping(rgb_u8, mapping: Mapping, strength: float):
    layers=_layers(mapping);t=_strength(strength)
    rgb=np.asarray(rgb_u8)
    if rgb.dtype != np.uint8 or rgb.ndim < 2 or rgb.shape[-1] != 3 or rgb.size == 0:
        raise ValueError("Expected nonempty uint8 RGB pixels")
    if t == 0:
        return rgb.copy()
    # Bound interpolation scratch memory even for multi-megapixel images.
    pixels=rgb.reshape(-1,3)
    output=np.empty(pixels.shape,np.uint8)
    for start in range(0,len(pixels),65536):
        stop=min(start+65536,len(pixels))
        result=_compose(layers,pixels[start:stop].astype(np.float32)/F(255),t)
        output[start:stop]=np.rint(np.clip(result,0,1)*F(255)).astype(np.uint8)
    return output.reshape(rgb.shape)


def cube_text(mapping: Mapping, strength: float, size=64) -> str:
    if type(size) is not int or size != 64:
        raise ValueError("Only LUT size64 is supported")
    layers=_layers(mapping);t=_strength(strength)
    axis=np.linspace(0,1,64,dtype=np.float32)
    rgb=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),axis=-1)
    combined=_compose(layers,rgb,t)
    # .cube row order: R changes fastest, then G, then B.
    rows=combined.transpose(2,1,0,3).reshape(-1,3)
    return 'TITLE "Reference Color"\nLUT_3D_SIZE 64\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'+''.join('%.9f %.9f %.9f\n'%tuple(row) for row in rows)
