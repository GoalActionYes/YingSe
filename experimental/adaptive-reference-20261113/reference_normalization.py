"""Training-only original-reference histogram task, never an inference branch."""
import torch
import torch.nn.functional as F


def distribution(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(3,128) or x.shape[0]<1 or not x.is_floating_point() or not torch.isfinite(x).all() or (x<0).any() or not torch.allclose(x.sum(-1),torch.ones_like(x[:,:,0]),atol=1e-5):raise ValueError('Expected normalized B3x128 distributions')


class ReferenceNormalization(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.GELU(),torch.nn.Linear(128,384))

    def forward(self,features,histogram):
        distribution(histogram)
        if features.ndim!=4 or features.shape[:2]!=(histogram.shape[0],128) or not torch.isfinite(features).all():raise ValueError('Expected shared B128HW features')
        x=torch.cat([features.mean((-2,-1)),histogram.flatten(1)],1)
        return F.softmax(self.layers(x).reshape(-1,3,128),dim=-1)


def histogram_loss(prediction,teacher):
    distribution(prediction);distribution(teacher)
    if prediction.shape!=teacher.shape:raise ValueError('Histogram batch mismatch')
    if teacher.requires_grad:raise ValueError('Teacher must be frozen')
    return ((prediction.add(1e-8).sqrt()-teacher.add(1e-8).sqrt()).square().sum(-1)).mean()
