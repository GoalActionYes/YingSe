"""One fixed staged paired-loss treatment, preserving the plain control loss."""
import torch.nn.functional as F
from paired_training import paired_delta_loss


def staged_loss(prediction,truth,paired):
    if type(paired) is not bool:raise ValueError('Expected explicit paired mode')
    rgb=F.l1_loss(prediction,truth);edge=F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],truth[:,:,1:]-truth[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],truth[:,:,:,1:]-truth[:,:,:,:-1]);delta=paired_delta_loss(prediction,truth)
    total=rgb+.1*edge
    if paired:total=total+.25*delta
    return dict(total=total,rgb=rgb,edge=edge,delta=delta)


def staged_gate(matched,original,paired):
    a=matched.get('proceed_to_independent_evaluation');b=original.get('proceed_to_independent_evaluation')
    if type(a) is not bool or type(b) is not bool or type(paired) is not bool:raise ValueError('Expected complete explicit gates')
    return dict(matched,matched_control_gate_passed=a,original_C10_gate_passed=b,paired_candidate=paired,proceed_to_independent_evaluation=paired and a and b)
