"""Fixed diagnostic transforms and probes; not an image-transfer engine."""
import numpy as np
import torch

OPERATIONS=('identity','warm','cool','saturation_high','saturation_low',
            'contrast_high','contrast_low','bright','dark','cool_shadows','warm_shadows')


def transform(rgb,operation):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape)<1 or operation not in OPERATIONS:
        raise ValueError('Expected nonempty uint8 RGB and a fixed operation')
    x=rgb.astype(np.float64)/255
    y=x@np.array([.2126,.7152,.0722])
    if operation=='identity':return rgb.copy()
    if operation in ('warm','cool'):x=x+np.array([.06,0,-.06])*(1 if operation=='warm' else -1)
    elif operation.startswith('saturation_'):x=y[...,None]+(x-y[...,None])*(1.4 if operation=='saturation_high' else .6)
    elif operation.startswith('contrast_'):x=.5+(x-.5)*(1.3 if operation=='contrast_high' else .7)
    elif operation in ('bright','dark'):x=x+(.08 if operation=='bright' else -.08)
    else:x=x+(2*y[...,None]-1)*np.array([.08,0,-.08])*(1 if operation=='cool_shadows' else -1)
    return np.rint(x.clip(0,1)*255).astype(np.uint8)


def appearance_statistics(features):
    if not isinstance(features,torch.Tensor) or features.ndim!=4 or min(features.shape)<1 or not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError('Expected finite nonempty floating BCHW features')
    flat=features.flatten(2);mean=flat.mean(-1);std=flat.std(-1,unbiased=False)
    gram=flat@flat.transpose(1,2)/flat.shape[-1];i,j=torch.triu_indices(flat.shape[1],flat.shape[1],device=flat.device)
    return mean,std,gram[:,i,j]


def group_folds(sources):
    if len(sources)<4 or len(sources)%2 or len(set(sources))!=len(sources):raise ValueError('Expected distinct even source groups')
    sources=sorted(sources);order=np.random.default_rng(20261114).permutation(len(sources))
    return {sources[int(index)]:int(i>=len(sources)//2) for i,index in enumerate(order)}


def ridge_predict(train,labels,test,classes):
    train,test=np.asarray(train,dtype=np.float64),np.asarray(test,dtype=np.float64);labels=np.asarray(labels)
    if train.ndim!=2 or test.ndim!=2 or min(train.shape)<1 or test.shape[1]!=train.shape[1] or labels.shape!=(len(train),) or not np.issubdtype(labels.dtype,np.integer) or type(classes)!=int or classes<2 or not np.isfinite(train).all() or not np.isfinite(test).all() or np.any((labels<0)|(labels>=classes)):
        raise ValueError('Invalid diagnostic ridge data')
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-6);normalizer=np.sqrt(train.shape[1])
    a=(train-mean)/scale/normalizer;b=(test-mean)/scale/normalizer
    kernel=a@a.T;kernel.flat[::len(train)+1]+=1
    alpha=np.linalg.solve(kernel,np.eye(classes)[labels]);scores=b@a.T@alpha
    if not np.isfinite(scores).all():raise ValueError('Nonfinite diagnostic prediction')
    return scores.argmax(1)
