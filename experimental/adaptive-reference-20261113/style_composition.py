"""Analytic spatial color composition; no pixel motion or generated structures."""
import torch
from style_probe import PARAM_SCALE


def _fields(rgb, person, code):
    if person.shape != (rgb.shape[0], 1, *rgb.shape[2:]) or code.shape != (rgb.shape[0], 2, 6):
        raise ValueError('Color composition shape mismatch')
    return (code[:, 0, :, None, None]*person+code[:, 1, :, None, None]*(1-person))*rgb.new_tensor(PARAM_SCALE)[None, :, None, None]


def _saturation(rgb, log_amount):
    gray = (rgb*rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(dim=1, keepdim=True)
    return rgb+(rgb-gray)*(log_amount.exp()-1)


def _gains(fields):
    return torch.stack((fields[:, 0]+fields[:, 2], fields[:, 2], fields[:, 1]+fields[:, 2]), dim=1).exp()


def _zero_preserving_power(value, exponent):
    # x**a has an infinite boundary derivative for 0<a<1. Evaluate a finite
    # surrogate below 1e-6 while preserving exact zero. With our exponent range
    # this branch cannot change any RGB8 output code, but avoids 0*Inf backward.
    positive = value.clamp_min(1e-6).pow(exponent)
    return torch.where(value > 0, positive, torch.zeros_like(positive))


def fallback_unobserved_roles(code, auto_person):
    observed = auto_person.mean(dim=(1, 2, 3)) >= .02
    foreground = torch.where(observed[:, None], code[:, 0], code[:, 1])
    return torch.stack((foreground, code[:, 1]), dim=1)


def compose_styles(rgb, person, source, reference):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('Expected NCHW RGB')
    for tensor, low, high in ((rgb, 0, 1), (person, 0, 1), (source, -1, 1), (reference, -1, 1)):
        if not torch.isfinite(tensor).all() or bool(((tensor < low) | (tensor > high)).any()):
            raise ValueError('Color composition requires bounded finite inputs')
    before, after = _fields(rgb, person, source), _fields(rgb, person, reference)
    # Keep neutral colors unbounded: clipping here breaks F_s(F_s^-1(y)) = y.
    neutral = (_zero_preserving_power(rgb, before[:, 5:6].exp())-before[:, 3:4])/_gains(before)
    neutral = _saturation(neutral, -before[:, 4:5])
    styled = (_saturation(neutral, after[:, 4:5])*_gains(after)+after[:, 3:4]).clamp(0, 1)
    return _zero_preserving_power(styled, (-after[:, 5:6]).exp())
