"""Fixed median-bandwidth centered Gaussian kernel ridge, train-only statistics."""
import numpy as np


def matrix(value):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=2 or min(value.shape)<1 or not np.isfinite(value).all():raise ValueError('Expected finite nonempty matrix')
    return value


def distances(a,b):
    return np.maximum(0,np.sum(a*a,axis=1)[:,None]+np.sum(b*b,axis=1)[None,:]-2*a@b.T)


def fit(x,y):
    x,y=matrix(x),matrix(y)
    if len(x)<2 or len(x)!=len(y):raise ValueError('Training rows differ or insufficient')
    mean=x.mean(0);center=x-mean;d=distances(center,center);bandwidth=float(np.median(d[np.triu_indices(len(x),1)]))
    if not bandwidth>1e-12:raise ValueError('Degenerate training bandwidth')
    kernel=np.exp(-d/(2*bandwidth));column=kernel.mean(0);grand=float(kernel.mean());kc=kernel-column[None,:]-kernel.mean(1)[:,None]+grand;lam=.1*float(np.trace(kc))/len(x);ym=y.mean(0)
    if not lam>0:raise ValueError('Degenerate centered kernel')
    dual=np.linalg.solve(kc+lam*np.eye(len(x)),y-ym)
    return dict(xmean=mean,center=center,bandwidth_squared=np.array(bandwidth),kernel_column_mean=column,kernel_grand_mean=np.array(grand),lambda_value=np.array(lam),dual=dual,ymean=ym)


def predict(model,x):
    x=matrix(x)
    if x.shape[1]!=model['center'].shape[1]:raise ValueError('Prediction feature width differs')
    kernel=np.exp(-distances(x-model['xmean'],model['center'])/(2*float(model['bandwidth_squared'])));kc=kernel-kernel.mean(1)[:,None]-model['kernel_column_mean'][None,:]+float(model['kernel_grand_mean']);out=kc@model['dual']+model['ymean']
    if not np.isfinite(out).all():raise ValueError('Nonfinite kernel prediction')
    return out
