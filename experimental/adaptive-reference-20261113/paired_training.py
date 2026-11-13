"""Deterministic same-content pairs and differentiable reference-response loss."""
from collections import defaultdict
import torch
import torch.nn.functional as F


def grouped_batches(rows,rng):
    groups=defaultdict(list)
    for index,row in enumerate(rows):groups[row['group_id']].append(index)
    if not groups or len(groups)%2:raise ValueError('Expected positive even group count')
    for indices in groups.values():
        if sorted(rows[i]['style_index'] for i in indices)!=list(range(4)) or len({rows[i]['content'] for i in indices})!=1:raise ValueError('Expected four references for the same content')
    names=list(groups);order=rng.permutation(len(names));styles={n:rng.permutation(groups[n]).tolist() for n in names}
    pairs=[styles[names[int(j)]][offset:offset+2] for offset in [0,2] for j in order]
    return [pairs[i]+pairs[i+1] for i in range(0,len(pairs),2)]


def paired_delta_loss(prediction,gt):
    if prediction.ndim!=4 or prediction.shape!=gt.shape or prediction.shape[0]!=4 or prediction.shape[1]!=3:raise ValueError('Expected two RGB content pairs')
    if not prediction.is_floating_point() or not gt.is_floating_point():raise ValueError('Paired differences require floating point')
    return F.l1_loss(prediction[1::2]-prediction[::2],gt[1::2]-gt[::2])
