"""Bounded synthetic styles and reference-only style-code regressors.

This is an identifiability experiment, not the product color-transfer algorithm.
Role order: person, other. Parameter order: red gain, blue gain, total gain,
offset, saturation, gamma (normalized coordinates scaled by PARAM_SCALE).
"""
import torch
from torch import nn

PARAM_SCALE = (.12, .12, .18, .045, .30, .15)


def render_style(rgb, person, normalized):
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or person.shape != (rgb.shape[0], 1, *rgb.shape[2:])
            or normalized.shape != (rgb.shape[0], 2, 6)):
        raise ValueError('Style image/mask/code shape mismatch')
    for tensor, low, high in ((rgb, 0, 1), (person, 0, 1), (normalized, -1, 1)):
        if not torch.isfinite(tensor).all() or bool(((tensor < low) | (tensor > high)).any()):
            raise ValueError('Style controls must be finite and bounded')
    code = normalized[:, 0, :, None, None]*person + normalized[:, 1, :, None, None]*(1-person)
    value = code*rgb.new_tensor(PARAM_SCALE)[None, :, None, None]
    gray = (rgb*rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(dim=1, keepdim=True)
    result = rgb+(rgb-gray)*(value[:, 4:5].exp()-1)
    gains = torch.stack((value[:, 0]+value[:, 2], value[:, 2], value[:, 1]+value[:, 2]), dim=1).exp()
    result = (result*gains+value[:, 3:4]).clamp(0, 1)
    return result.pow((-value[:, 5:6]).exp())


def draw_style_codes(count, generator):
    if count < 1:
        raise ValueError('Expected positive style count')
    codes = torch.rand(count, 2, 6, generator=generator)*2-1
    mode = torch.rand(count, generator=generator)
    global_mask = (mode >= .2) & (mode < .5)
    codes[global_mask, 1] = codes[global_mask, 0]
    codes[mode < .2] = 0
    return codes


def masked_parameter_mse(prediction, truth, availability):
    if prediction.shape != truth.shape or availability.shape != prediction.shape[:2]:
        raise ValueError('Parameter supervision shape mismatch')
    error = (prediction-truth).square().mean(dim=-1)
    return (error*availability).sum()/availability.sum().clamp_min(1)


class StyleRegressor(nn.Module):
    def __init__(self, stats_dim, semantic_dim, use_semantic):
        super().__init__()
        self.use_semantic = use_semantic
        self.stats_dim, self.semantic_dim = stats_dim, semantic_dim
        self.stats = nn.Sequential(nn.Linear(stats_dim, 64), nn.SiLU())
        if use_semantic:
            self.semantic = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, 128), nn.SiLU())
        width = 192 if use_semantic else 64
        self.predictor = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 12), nn.Tanh())

    def forward(self, stats, semantic):
        features = self.stats(stats)
        if self.use_semantic:
            features = torch.cat((features, self.semantic(semantic)), dim=1)
        return self.predictor(features).reshape(-1, 2, 6)
