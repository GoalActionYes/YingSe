"""Bounded color-only residuals from confidence-weighted patch correspondences."""
import torch
import torch.nn.functional as F


def _unit(tensor):
    return torch.is_floating_point(tensor) and bool(torch.isfinite(tensor).all()) and bool(((tensor>=0)&(tensor<=1)).all())


def match_anchors(target_features,reference_features,reference_colors):
    if (target_features.ndim!=2 or reference_features.ndim!=2 or target_features.shape[1]!=reference_features.shape[1]
            or min(target_features.shape[0],reference_features.shape[0])<1
            or reference_colors.shape!=(reference_features.shape[0],3)
            or not torch.isfinite(target_features).all() or not torch.isfinite(reference_features).all()
            or not _unit(reference_colors)):
        raise ValueError('Invalid patch feature/color arrays')
    a,b=F.normalize(target_features,dim=-1),F.normalize(reference_features,dim=-1)
    similarity=a@b.T
    values,indices=similarity.topk(min(4,b.shape[0]),dim=1)
    weights=(values/.05).softmax(dim=1)
    # Valid RGB convex combinations can exceed1 by a float32 ulp at white.
    # Validate source colors above, then restore the mathematical unit bound.
    anchors=(reference_colors[indices]*weights[...,None]).sum(dim=1).clamp(0,1)
    reverse=similarity.argmax(dim=0)
    cycle=(a*a[reverse[indices[:,0]]]).sum(dim=-1)
    confidence=((values[:,0]-.65)/.20).clamp(0,1)*((cycle-.85)/.15).clamp(0,1)
    return anchors,confidence


def _box(x,size):
    return F.avg_pool2d(x,size,stride=1,padding=size//2,count_include_pad=False)


def guided_upsample(guide,field):
    if guide.ndim!=4 or field.ndim!=4 or guide.shape[:2]!=(1,3) or field.shape[:2]!=(1,3):
        raise ValueError('Expected one RGB guidance image and three-channel field')
    if not _unit(guide) or not torch.isfinite(field).all(): raise ValueError('Invalid guided upsample inputs')
    low=F.interpolate(guide,size=field.shape[2:],mode='bilinear',align_corners=False)
    # Color guided filter at low resolution; upsample affine coefficients, not pixels.
    mi,mp=_box(low,5),_box(field,5)
    covariance=torch.stack([_box(low[:,i:i+1]*low[:,j:j+1],5)-mi[:,i:i+1]*mi[:,j:j+1]
        for i in range(3) for j in range(3)],dim=-1).reshape(*low.shape[2:],3,3)
    cross=torch.stack([_box(low[:,i:i+1]*field[:,j:j+1],5)-mi[:,i:i+1]*mp[:,j:j+1]
        for i in range(3) for j in range(3)],dim=-1).reshape(*low.shape[2:],3,3)
    coefficient=torch.linalg.solve(covariance+torch.eye(3,device=low.device,dtype=low.dtype)*.001,cross)
    # coefficient[input_channel, output_channel] at each low-resolution site.
    a=coefficient.permute(2,3,0,1).reshape(1,9,*low.shape[2:])
    b=mp-torch.einsum('bichw,bihw->bchw',a.reshape(1,3,3,*low.shape[2:]),mi)
    a=F.interpolate(_box(a,5),size=guide.shape[2:],mode='bilinear',align_corners=False).reshape(1,3,3,*guide.shape[2:])
    b=F.interpolate(_box(b,5),size=guide.shape[2:],mode='bilinear',align_corners=False)
    return torch.einsum('bichw,bihw->bchw',a,guide)+b


def color_from_anchors(base,anchors,confidence,*,spatial=False,strength=1.):
    if (base.ndim!=4 or base.shape[:2]!=(1,3) or anchors.ndim!=4 or anchors.shape[:2]!=(1,3)
            or confidence.shape!=(1,1,*anchors.shape[2:]) or not all(_unit(x) for x in (base,anchors,confidence))
            or not isinstance(strength,(int,float)) or isinstance(strength,bool) or not 0<=strength<=1):
        raise ValueError('Invalid color/mask inputs or strength')
    diagnostics=dict(mean_confidence=float(confidence.mean()),spatial=spatial,strength=strength)
    if strength==0 or diagnostics['mean_confidence']<.1:
        return base.clone(),dict(diagnostics,fallback=True)
    colors=F.adaptive_avg_pool2d(base,anchors.shape[2:])[0].flatten(1).T
    target=anchors[0].flatten(1).T
    weight=confidence.flatten()
    design=torch.cat((colors-.5,torch.ones_like(colors[:,:1])),dim=1)
    matrix=design.T@(design*weight[:,None])/weight.sum()
    rhs=design.T@((target-colors)*weight[:,None])/weight.sum()
    penalty=torch.diag(base.new_tensor([.02,.02,.02,.002]))
    coefficient=torch.linalg.solve(matrix+penalty,rhs)
    coefficient=torch.cat((coefficient[:3].clamp(-.5,.5),coefficient[3:].clamp(-.12,.12)),dim=0)
    pixel_design=torch.cat((base-.5,torch.ones_like(base[:,:1])),dim=1)
    delta=torch.einsum('bihw,ic->bchw',pixel_design,coefficient).clamp(-.12,.12)
    result=(base+.5*strength*delta).clamp(0,1)
    diagnostics.update(fallback=False,global_coefficient=coefficient.tolist())
    if spatial:
        remaining=(anchors-F.adaptive_avg_pool2d(result,anchors.shape[2:])).clamp(-.12,.12)
        smooth_weight=_box(confidence,3)
        field=_box(remaining*confidence,3)/smooth_weight.clamp_min(1e-6)
        field=field*smooth_weight
        local=guided_upsample(base,field).clamp(-.12,.12)
        result=(result+.5*strength*local).clamp(0,1)
    if not torch.isfinite(result).all(): raise ValueError('Nonfinite color residual output')
    return result,diagnostics


def choose_blend(first,second,anchors,confidence):
    scores=[]
    for fraction in (0,.25,.5,.75,1):
        patch=F.adaptive_avg_pool2d(first*(1-fraction)+second*fraction,anchors.shape[2:])
        score=float(((patch-anchors).square().mean(dim=1,keepdim=True)*confidence).sum()/confidence.sum().clamp_min(1e-6))
        scores.append(dict(fraction=fraction,score=score))
    fraction=0 if confidence.mean()<.1 else min(scores,key=lambda row:row['score'])['fraction']
    return first*(1-fraction)+second*fraction,dict(selected_fraction=fraction,curve=scores)
