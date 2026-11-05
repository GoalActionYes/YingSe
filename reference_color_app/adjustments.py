"""Independent global grading controls; not Jianying parameter equivalence."""
import copy
import numbers
import numpy as np
from .contracts import Mapping
from .lut import _compose, _layers, _strength

BASIC = ('temperature','tint','saturation','vibrance','brightness','contrast',
         'highlights','shadows','whites','blacks','brilliance')
COLORS = ('red','orange','yellow','green','cyan','blue','purple','magenta')
CENTERS = np.array([0,30,60,120,180,240,275,315],np.float32)/360


def _number(value, low=-100, high=100):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not np.isfinite(value) or not low<=value<=high:
        raise ValueError('调色参数必须是范围内的有限数字')
    return float(value)


def normalize(state=None):
    if state is None:state={}
    if not isinstance(state,dict) or set(state)-{'version','basic','hsl','curves'}:
        raise ValueError('调色设置格式无效')
    if type(state.get('version',1)) is not int or state.get('version',1)!=1:
        raise ValueError('不支持此调色设置版本')
    result={'version':1}
    for group,allowed in [('basic',BASIC),('hsl',COLORS),('curves',('rgb','r','g','b'))]:
        source=state.get(group,{})
        if not isinstance(source,dict) or set(source)-set(allowed)-{'enabled'} or type(source.get('enabled',True)) is not bool:
            raise ValueError('调色分组格式无效')
        target={'enabled':source.get('enabled',True)}
        for key in allowed:
            if group=='basic':target[key]=_number(source.get(key,0))
            elif group=='hsl':
                values=source.get(key,[0,0,0])
                if not isinstance(values,list) or len(values)!=3:raise ValueError('HSL 需要三个数字')
                target[key]=[_number(v) for v in values]
            else:
                points=source.get(key,[[0,0],[1,1]])
                if not isinstance(points,list) or not 2<=len(points)<=16:raise ValueError('曲线需要 2 至 16 个点')
                clean=[]
                for point in points:
                    if not isinstance(point,list) or len(point)!=2:raise ValueError('曲线点格式无效')
                    clean.append([_number(v,0,1) for v in point])
                if clean[0][0]!=0 or clean[-1][0]!=1 or any(b[0]-a[0]<.001 for a,b in zip(clean,clean[1:])):
                    raise ValueError('曲线横坐标必须从 0 到 1 严格递增')
                target[key]=clean
        result[group]=target
    return result


def is_neutral(state):
    base=normalize()
    return all(not state[g]['enabled'] or all(state[g][k]==base[g][k] for k in base[g] if k!='enabled') for g in ('basic','hsl','curves'))


def _hsl(rgb):
    maximum=rgb.max(axis=-1);minimum=rgb.min(axis=-1);delta=maximum-minimum
    light=(maximum+minimum)*.5
    saturation=np.divide(delta,1-np.abs(2*light-1),out=np.zeros_like(delta),where=(delta>1e-7)&(light>0)&(light<1))
    safe=np.where(delta>1e-7,delta,1)
    r,g,b=np.moveaxis(rgb,-1,0)
    hue=np.where(maximum==r,((g-b)/safe)%6,np.where(maximum==g,(b-r)/safe+2,(r-g)/safe+4))/6
    return np.where(delta>1e-7,hue,0),saturation,light


def _from_hsl(h,s,l):
    c=(1-np.abs(2*l-1))*s
    hp=(h%1)*6;x=c*(1-np.abs(hp%2-1));zero=np.zeros_like(c)
    options=np.stack([np.stack(v,axis=-1) for v in [(c,x,zero),(x,c,zero),(zero,c,x),(zero,x,c),(x,zero,c),(c,zero,x)]])
    index=np.floor(hp).astype(int)%6
    selected=np.take_along_axis(options,index[None,...,None],axis=0)[0]
    return selected+(l-c/2)[...,None]


def transform(rgb,state=None):
    state=normalize(state)
    rgb=np.asarray(rgb,np.float32)
    if is_neutral(state):return rgb.copy()
    x=rgb.copy();p=state['basic']
    if p['enabled']:
        v={k:p[k]/100 for k in BASIC}
        x*=np.exp2(np.array([v['temperature']*.35,-v['tint']*.2,-v['temperature']*.35],np.float32))
        x[...,0]*=np.exp2(v['tint']*.1);x[...,2]*=np.exp2(v['tint']*.1)
        y=np.clip(x@np.array([.2126,.7152,.0722],np.float32),0,1)
        low=(1-y)**2;high=y**2;mid=4*y*(1-y)
        offset=v['brightness']*.35+v['shadows']*.25*low+v['highlights']*.25*high
        offset+=v['blacks']*.2*low**3+v['whites']*.2*high**3+v['brilliance']*.2*mid*(1-y)
        x+=offset[...,None]
        x=(x-.5)*np.exp2(v['contrast'])+.5
        gray=x@np.array([.2126,.7152,.0722],np.float32)
        chroma=np.clip(x.max(-1)-x.min(-1),0,1)
        factor=np.maximum(0,1+v['saturation']+v['vibrance']*(1-chroma)**2)
        x=gray[...,None]+(x-gray[...,None])*factor[...,None]
        x=np.clip(x,0,1)
    p=state['hsl']
    if p['enabled'] and any(any(p[k]) for k in COLORS):
        hue,sat,light=_hsl(x);delta_h=np.zeros_like(hue);delta_s=np.zeros_like(hue);delta_l=np.zeros_like(hue)
        for key,center in zip(COLORS,CENTERS):
            distance=np.abs((hue-center+.5)%1-.5)
            weight=np.maximum(0,1-distance/(50/360))**2
            a,b,c=p[key];delta_h+=weight*a/100*(30/360);delta_s+=weight*b/100;delta_l+=weight*c/100*.3
        x=_from_hsl(hue+delta_h,np.clip(sat*(1+delta_s),0,1),np.clip(light+delta_l*np.minimum(1,sat*5),0,1))
    p=state['curves']
    if p['enabled']:
        points=np.asarray(p['rgb']);x=np.interp(x,points[:,0],points[:,1]).astype(np.float32)
        for i,key in enumerate(('r','g','b')):
            points=np.asarray(p[key]);x[...,i]=np.interp(x[...,i],points[:,0],points[:,1])
    return np.clip(x,0,1).astype(np.float32)


def compose(mapping,strength,state=None):
    """Return (effective mapping, strength); neutral values preserve old path."""
    state=normalize(state);strength=float(_strength(strength))
    if is_neutral(state):return mapping,strength
    axis=np.linspace(0,1,64,dtype=np.float32)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    graded=transform(_compose(_layers(mapping),grid,strength),state)
    return Mapping(np.stack([graded,graded]),dict(mapping.diagnostics,manual_adjustments=copy.deepcopy(state),manual_algorithm_version=1)),1.0
