"""GT-assisted byte-domain capacity helpers; never an inference policy."""
import numpy as np


def _validate(a,b):
    for x in [a,b]:
        if not isinstance(x,np.ndarray) or x.dtype!=np.uint8 or x.ndim!=3 or x.shape[-1]!=3 or min(x.shape)<1:raise ValueError('Expected nonempty uint8 RGB arrays')
    if a.shape!=b.shape:raise ValueError('RGB shapes differ')


def blend_codes(a,b,step):
    _validate(a,b)
    if type(step)!=int or not 0<=step<=100:raise ValueError('Expected integer share0..100')
    numerator=(100-step)*a.astype(np.int32)+step*b.astype(np.int32);q,remainder=np.divmod(numerator,100)
    return (q+((remainder>50)|((remainder==50)&((q%2)==1)))).astype(np.uint8)


def rgb_oracle(a,b,truth):
    _validate(a,b);_validate(a,truth);truth=truth.astype(np.int16)
    scores=[float(np.abs(blend_codes(a,b,k).astype(np.int16)-truth).mean()) for k in range(101)]
    return scores,int(np.argmin(scores))
