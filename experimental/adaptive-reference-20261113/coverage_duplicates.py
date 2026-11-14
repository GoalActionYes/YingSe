"""Fixed exact and DCT perceptual screening, not semantic identity proof."""
import hashlib
import numpy as np
from PIL import Image


def _rgb(array):
    a=np.asarray(array)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<1:
        raise ValueError('Expected positive-size RGB8 array')
    return a


def pixel_digest(array):
    a=_rgb(array)
    return hashlib.sha256(np.array(a.shape,dtype='<i8').tobytes()+a.tobytes()).hexdigest()


def perceptual_hash(array):
    gray=np.array(Image.fromarray(_rgb(array)).convert('L').resize((32,32),Image.Resampling.BILINEAR),dtype=np.float64)
    if np.ptp(gray)==0:return 0
    # Subtracting DC improves numerical stability without changing AC coefficients.
    gray-=gray.mean();n=np.arange(32,dtype=np.float64);k=n[:,None]
    dct=np.sqrt(2/32)*np.cos(np.pi*(n+.5)*k/32);dct[0]/=np.sqrt(2)
    low=(dct@gray@dct.T)[:8,:8].ravel()[1:];bits=low>np.median(low)
    return sum(int(bit)<<i for i,bit in enumerate(bits))


def near_crossings(train,heldout):
    return [dict(train=a['path'],heldout=b['path'],distance=distance)
            for a in train for b in heldout if (distance:=(a['hash']^b['hash']).bit_count())<=6]
