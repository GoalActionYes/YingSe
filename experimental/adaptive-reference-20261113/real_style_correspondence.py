"""Tie-aware offline correspondence measurements, not an inference model."""
import itertools
import numpy as np


def correspondence(cost):
    cost=np.asarray(cost,dtype=np.float64)
    if cost.shape!=(4,4) or not np.isfinite(cost).all() or (cost<0).any():raise ValueError('Expected finite nonnegative 4x4 costs')
    ties=np.isclose(cost,cost.min(1)[:,None],rtol=1e-10,atol=1e-12)
    row=float(np.mean(ties.diagonal()/ties.sum(1)))
    permutations=list(itertools.permutations(range(4)));values=np.array([cost[np.arange(4),p].sum() for p in permutations])
    selected=np.flatnonzero(np.isclose(values,values.min(),rtol=1e-10,atol=1e-12));diagonal=float(np.trace(cost)/4);other=float(cost[~np.eye(4,dtype=bool)].mean())
    return dict(row_top1=row,assignment_fraction_correct=float(np.mean([sum(i==p for i,p in enumerate(permutations[j]))/4 for j in selected])),identity_optimum_fraction=float((0 in selected)/len(selected)),optimal_assignments=len(selected),diagonal_mean=diagonal,offdiagonal_mean=other,margin=other-diagonal)


def gt_differences(images):
    if not isinstance(images,np.ndarray) or images.dtype!=np.uint8 or images.ndim!=4 or images.shape[0]!=4 or images.shape[-1]!=3 or min(images.shape)<1:raise ValueError('Expected four nonempty uint8 RGB images')
    rgb=images.astype(np.float64)
    return [float(np.abs(rgb[i]-rgb[j]).mean()) for i,j in itertools.combinations(range(4),2)]
