"""Constructive GT-assisted fit; not an optimal bound or inference method."""
import numpy as np


def design(rgb,probability,regional):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB8')
    if probability.shape!=rgb.shape[:2] or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():raise ValueError('Invalid probability')
    x=np.column_stack([rgb.reshape(-1,3).astype(float)/255,np.ones(rgb.shape[0]*rgb.shape[1])])
    return np.column_stack([x,x*probability.reshape(-1,1)]) if regional else x


def fit(rgb,probability,baseline,truth,regional):
    x=design(rgb,probability,regional)
    if baseline.shape!=rgb.shape or truth.shape!=rgb.shape or not np.isfinite(baseline).all() or not np.isfinite(truth).all():raise ValueError('Invalid float images')
    latent=np.arctanh(np.clip((truth-baseline)/.1,-.99,.99)).reshape(-1,3)
    gram=x.T@x/len(x);gram.flat[::len(gram)+1]+=1e-5
    coefficients=np.linalg.solve(gram,x.T@latent/len(x));residual=.1*np.tanh((x@coefficients).reshape(rgb.shape))
    if not np.isfinite(coefficients).all() or np.abs(residual).max()>.1:raise ValueError('Fit failed')
    return baseline+residual,coefficients,float(np.abs(residual).max())
