"""Trainable style backbone, independent of the frozen RGB-only mask network."""
import copy
import torch
import torch.nn.functional as F
from style_features import StyleFeatures, role_weights
from style_probe import StyleRegressor


class PixelFeatures(StyleFeatures):
    """Expose the automatic native probability without rerunning the backbone.

    The classifier hook captures only RGB-derived logits. Extraction is sequential
    in these research scripts; concurrent callers need their own instance.
    """
    def __init__(self, device='cuda:0'):
        super().__init__(device)
        self._native_logits = None
        self._hook = self.model.classifier.register_forward_hook(self._capture_logits)

    def _capture_logits(self, module, inputs, output):
        self._native_logits = output.detach()

    @torch.no_grad()
    def extract(self, rgb):
        result = super().extract(rgb)
        result['native_person'] = self._native_logits.softmax(dim=1)[:, 15:16]
        return result


def style_moments(features, probability):
    if probability.shape[-2:] != features.shape[-2:]:
        probability = F.interpolate(probability, size=features.shape[-2:], mode='bilinear', align_corners=False)
    weights = role_weights(probability)
    mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    mean = torch.einsum('bkhw,bchw->bkc', weights, features)/mass[:, :, None]
    second = torch.einsum('bkhw,bchw->bkc', weights, features.square())/mass[:, :, None]
    variance = (second-mean.square()).clamp_min(0)
    safe_std = (variance+1e-6).sqrt()
    # Exact frozen forward moments, finite derivative at zero variance.
    std = safe_std+(variance.detach().sqrt()-safe_std.detach())
    return torch.cat((mean.sign()*mean.abs().log1p(), std.log1p()), dim=-1).flatten(1)


class PixelStyleEncoder(torch.nn.Module):
    def __init__(self, frozen_backbone, feature_channels=960, stats_dim=42):
        super().__init__()
        self.backbone = copy.deepcopy(frozen_backbone)
        self.feature_channels = feature_channels
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self.head = StyleRegressor(stats_dim, feature_channels*6, use_semantic=True)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained running statistics with the small augmented batch.
        for module in self.backbone.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, rgb, stats, native_person):
        normalized = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        features = self.backbone(normalized)['out']
        if features.shape[1] != self.feature_channels:
            raise ValueError('Unexpected style-backbone channel count')
        return self.head(stats, style_moments(features, native_person.detach()))
