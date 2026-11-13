"""Frozen pretrained grading features, with new automatic regional pooling."""
import torch
import torch.nn.functional as F
from canonical_model import CanonicalModel
from style_features import StyleFeatures, pooled_moments, role_weights


def project_regions(features, person, projection):
    if (features.ndim != 4 or person.ndim != 4 or person.shape[:2] != (features.shape[0],1)
            or not torch.isfinite(features).all() or not torch.isfinite(person).all()
            or bool(((person<0)|(person>1)).any())):
        raise ValueError('Expected finite feature map and unit automatic probability')
    probability = F.interpolate(person,size=features.shape[2:],mode='bilinear',align_corners=False)
    means,_ = pooled_moments(features,role_weights(probability))
    b,roles,c = means.shape
    vectors = F.normalize(projection(means.reshape(b*roles,c)),dim=-1)
    return vectors.reshape(b,-1)


class CanonicalStyleFeatures(StyleFeatures):
    def __init__(self,device='cuda'):
        super().__init__(device)
        canonical = CanonicalModel(device=str(self.device))
        self.encoder = canonical.model.Embedding_Net
        self.encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        self.provenance.update(grading_features=canonical.provenance,
            regional_pooling='New experiment: global/person/other masked means before author projection; not author regional method',
            semantic_dimensions=192)

    @torch.no_grad()
    def extract(self,rgb):
        result = super().extract(rgb)
        features = self.encoder.net.backbone(self.encoder.downsample(rgb))
        result['semantic'] = project_regions(features,result['person'],self.encoder.net.channel_fitting).detach()
        return result
