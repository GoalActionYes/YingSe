"""Reference-derived convex choice among three dynamically computed global looks."""
import torch
import torch.nn.functional as F
from dense_anchors import _unit


def choose_global(candidates,anchors,confidence):
    if (len(candidates)!=3 or candidates[0].ndim!=4 or candidates[0].shape[:2]!=(1,3)
            or any(x.shape!=candidates[0].shape or not _unit(x) for x in candidates)
            or anchors.ndim!=4 or anchors.shape[:2]!=(1,3) or confidence.shape!=(1,1,*anchors.shape[2:])
            or not _unit(anchors) or not _unit(confidence)):
        raise ValueError('Expected three normalized global candidates and matching color anchors')
    mean=float(confidence.mean())
    if mean<.1:
        return candidates[1].clone(),dict(weights=[0.,1.,0.],mean_confidence=mean,fallback=True)
    patches=[F.adaptive_avg_pool2d(x,anchors.shape[2:]).flatten().double() for x in candidates]
    target=anchors.flatten().double(); weight=confidence.flatten().double().repeat(3)
    weight=weight/weight.sum()
    v=torch.stack((patches[1]-patches[0],patches[2]-patches[0]),dim=1)
    matrix=v.T@(v*weight[:,None]); rhs=v.T@((target-patches[0])*weight)
    # The optimum on a convex triangle lies inside or on one of its three edges.
    choices=[v.new_tensor([0.,1.,0.]),v.new_tensor([1.,0.,0.]),v.new_tensor([0.,0.,1.])]
    if float(torch.linalg.det(matrix))>1e-15:
        q=torch.linalg.solve(matrix,rhs)
        if bool((q>=0).all()) and float(q.sum())<=1:
            choices.append(torch.stack((1-q.sum(),q[0],q[1])))
    for i,j in ((0,1),(0,2),(1,2)):
        direction=patches[j]-patches[i]
        den=(direction.square()*weight).sum()
        fraction=(((target-patches[i])*direction*weight).sum()/den).clamp(0,1) if den>1e-20 else den.new_zeros(())
        q=v.new_zeros(3); q[i]=1-fraction; q[j]=fraction; choices.append(q)
    scores=[float(((sum(x*w for x,w in zip(patches,q))-target).square()*weight).sum()) for q in choices]
    selected=min(range(len(scores)),key=scores.__getitem__)
    q=choices[selected].to(candidates[0].dtype)
    result=sum(x*w for x,w in zip(candidates,q)).clamp(0,1)
    return result,dict(weights=q.tolist(),mean_confidence=mean,fallback=False,anchor_mse=scores[selected],
        candidate_mse=scores[:3],optimization='Convex quadratic interior/three edges, no expert target')
