"""RGB-only frozen features; human labels are never accepted by this interface."""
import torch
import torch.nn.functional as F
from person_segmenter import PersonSegmenter


def pooled_moments(features, weights):
    if (features.ndim != 4 or weights.ndim != 4 or features.shape[0] != weights.shape[0]
            or features.shape[2:] != weights.shape[2:]):
        raise ValueError('Feature/region shape mismatch')
    mass = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
    mean = torch.einsum('bkhw,bchw->bkc', weights, features)/mass[:, :, None]
    second = torch.einsum('bkhw,bchw->bkc', weights, features.square())/mass[:, :, None]
    return mean, (second-mean.square()).clamp_min(0).sqrt()


def role_weights(person):
    return torch.cat((torch.ones_like(person), person, 1-person), dim=1)


def color_statistics(rgb, person):
    weights = role_weights(person)
    mean, std = pooled_moments(rgb, weights)
    mass = weights.sum(dim=(-1, -2)).clamp_min(1e-6)
    second = torch.einsum('bkhw,bchw,bdhw->bkcd', weights, rgb, rgb)/mass[:, :, None, None]
    covariance = second-mean[:, :, :, None]*mean[:, :, None, :]
    triangle = torch.triu_indices(3, 3, device=rgb.device)
    cov = covariance[:, :, triangle[0], triangle[1]]
    saturation = rgb.amax(dim=1, keepdim=True)-rgb.amin(dim=1, keepdim=True)
    sat_mean, _ = pooled_moments(saturation, weights)
    area = weights.mean(dim=(-1, -2))[:, :, None]
    return torch.cat(((mean-.5)*2, std*4, cov*16, sat_mean*4, area), dim=-1).flatten(1)


class StyleFeatures:
    def __init__(self, device='cuda'):
        self.segmenter = PersonSegmenter(device=device)
        self.model = self.segmenter.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.provenance = dict(self.segmenter.provenance,
                               preprocessing='RGB square resize224 before extraction; ImageNet mean/std; frozen eval',
                               limitation='224 inference differs from official segmentation preset resize520; quality not established')

    @torch.no_grad()
    def extract(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 224, 224) or rgb.device != self.device:
            raise ValueError('Expected normalized RGB batch [N,3,224,224] on configured device')
        if not torch.isfinite(rgb).all() or bool(((rgb < 0) | (rgb > 1)).any()):
            raise ValueError('RGB input outside finite unit range')
        normalized = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        features = self.model.backbone(normalized)['out']
        logits = self.model.classifier(features)
        native_person = logits.softmax(dim=1)[:, self.segmenter.label:self.segmenter.label+1]
        pixel_logits = F.interpolate(logits, size=rgb.shape[2:], mode='bilinear', align_corners=False)
        person = pixel_logits.softmax(dim=1)[:, self.segmenter.label:self.segmenter.label+1]
        mean, std = pooled_moments(features, role_weights(native_person))
        semantic = torch.cat((mean.sign()*mean.abs().log1p(), std.log1p()), dim=-1).flatten(1)
        stats = color_statistics(rgb, person)
        return {'stats': stats.detach(), 'semantic': semantic.detach(), 'person': person.detach()}
