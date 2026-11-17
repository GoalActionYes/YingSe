"""Pooled cross-fitting of two observed transformations, diagnostic GT only."""
import numpy as np
from reference_transfer_probe import crossfit_transfer


def pooled_crossfit(xa,ya,xb,yb,mask):
    if xa.shape!=xb.shape or ya.shape!=yb.shape or xa.shape[0]!=ya.shape[0]:raise ValueError('Expected equal pixel support for both images')
    x=np.concatenate([xa,xb]);y=np.concatenate([ya,yb]);m=np.concatenate([mask,mask])
    fit=crossfit_transfer(x,y,x,m)
    n=len(xa)
    return dict(fold_coefficients=fit['fold_coefficients'],a_crossfit=fit['reference_crossfit'][:n],b_crossfit=fit['reference_crossfit'][n:])
