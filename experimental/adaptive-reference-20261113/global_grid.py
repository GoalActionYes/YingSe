"""Research-only spatial-output constraint; condition still uses both images."""
import torch
from pairwise_context import pairwise_grid
from training_checks import finite_output


def collapse_grid(grid):
    if not isinstance(grid,torch.Tensor) or grid.ndim!=5 or grid.shape[0]<1 or tuple(grid.shape[1:])!=(13,8,16,16) or not grid.is_floating_point() or not torch.isfinite(grid).all() or not (grid[:,12]>0).all():raise ValueError('Expected finite native grid with positive alpha')
    return grid.mean((-2,-1),keepdim=True)


def projected_forward(model,api,content,reference,global_output=False):
    if type(global_output)!=bool:raise ValueError('Expected explicit projection flag')
    grid=pairwise_grid(model,api,content,reference)
    if global_output:grid=collapse_grid(grid)
    raw=model.apply_coeffs(model.sampler(grid,content),content)
    return finite_output(raw,content.shape),grid
