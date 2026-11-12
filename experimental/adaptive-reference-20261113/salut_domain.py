"""A fixed author-LUT inverse approximation, NOT camera-Log recovery.

Inverse fitting sees the forward table only, never photos or evaluation outputs.
Roundtrip residuals may reflect gamut clipping, inverse-grid approximation or a
numerical solve failure; they do not prove a color is mathematically unreachable.
"""
import numpy as np
import torch
import torch.nn.functional as F
from salut_interpolation import trilinear


def _sample_inverse(inverse, rgb):
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all()
            or bool(((rgb < 0) | (rgb > 1)).any())):
        raise ValueError('Expected finite RGB NCHW in [0,1]')
    if (inverse.ndim != 4 or inverse.shape[0] != 3 or len(set(inverse.shape[1:])) != 1
            or not torch.isfinite(inverse).all() or bool(((inverse < 0) | (inverse > 1)).any())):
        raise ValueError('Inverse lattice must contain finite bounded RGB coordinates')
    grid = (rgb.permute(0, 2, 3, 1)*2-1).unsqueeze(1)
    # The exact convex combination is bounded, but CUDA accumulation can exceed
    # 1 by one float32 ULP. Restore this analytic invariant AFTER validating the
    # source lattice; do not hide genuinely invalid input/table coordinates.
    return F.grid_sample(inverse.unsqueeze(0).expand(rgb.shape[0], -1, -1, -1, -1), grid,
                         align_corners=True, mode='bilinear', padding_mode='border').squeeze(2).clamp(0, 1)


@torch.inference_mode()
def fit_inverse(forward, dim=33, steps=20):
    if (forward.ndim != 4 or forward.shape[0] != 3 or len(set(forward.shape[1:])) != 1
            or not torch.isfinite(forward).all() or forward.dtype != torch.float32):
        raise ValueError('Expected finite float32 [3,D,D,D] forward LUT')
    if dim < 3 or steps < 1:
        raise ValueError('Invalid inverse grid or iteration count')
    device = forward.device
    axis = torch.linspace(0, 1, dim, device=device)
    b, g, r = torch.meshgrid(axis, axis, axis, indexing='ij')
    target = torch.stack((r, g, b)).reshape(1, 3, dim, dim*dim)
    indices = torch.arange(forward.shape[-1], device=device)
    neutral = forward[:, indices, indices, indices].cpu().numpy()
    log_axis = np.linspace(0, 1, forward.shape[-1])
    initial = [np.interp(target[0, c].cpu().numpy(), np.maximum.accumulate(neutral[c]), log_axis)
               for c in range(3)]
    x = torch.from_numpy(np.stack(initial)).unsqueeze(0).to(device=device, dtype=torch.float32)
    # A neutral seed can land on a clipped plateau with zero derivatives even
    # when an exact inverse exists. Search the ENTIRE fixed forward lattice;
    # this initialization uses no photo data. Bound distance-matrix memory.
    source_axis = torch.linspace(0, 1, forward.shape[-1], device=device)
    sb, sg, sr = torch.meshgrid(source_axis, source_axis, source_axis, indexing='ij')
    coordinates = torch.stack((sr, sg, sb)).reshape(3, -1).transpose(0, 1)
    colors = forward.reshape(3, -1).transpose(0, 1).contiguous()
    desired = target[0].reshape(3, -1).transpose(0, 1).contiguous()
    nearest_parts = []
    for start in range(0, desired.shape[0], 384):
        distance = torch.cdist(desired[start:start+384].unsqueeze(0), colors.unsqueeze(0))
        nearest_parts.append(coordinates[distance[0].argmin(dim=1)])
    nearest = torch.cat(nearest_parts).transpose(0, 1).reshape_as(x)
    prefer_nearest = (trilinear(forward, nearest)-target).square().sum(dim=1, keepdim=True) < (trilinear(forward, x)-target).square().sum(dim=1, keepdim=True)
    x = torch.where(prefer_nearest, nearest, x)
    eye = torch.eye(3, device=device)[None]*1e-6
    for _ in range(steps):
        value = trilinear(forward, x)
        residual = value-target
        derivatives = []
        for channel in range(3):
            positive, negative = x.clone(), x.clone()
            positive[:, channel] = (positive[:, channel]+.001).clamp(0, 1)
            negative[:, channel] = (negative[:, channel]-.001).clamp(0, 1)
            width = (positive-negative)[:, channel:channel+1].clamp_min(1e-8)
            derivatives.append((trilinear(forward, positive)-trilinear(forward, negative))/width)
        jacobian = torch.stack(derivatives, dim=-1).permute(0, 2, 3, 1, 4).reshape(-1, 3, 3)
        residual_flat = residual.permute(0, 2, 3, 1).reshape(-1, 3, 1)
        transpose = jacobian.transpose(1, 2)
        delta = torch.linalg.solve(transpose@jacobian+eye, transpose@residual_flat)
        delta = delta.reshape(1, dim, dim*dim, 3).permute(0, 3, 1, 2).clamp(-.08, .08)
        best_error = residual.square().sum(dim=1, keepdim=True)
        best_x = x
        for scale in (1., .5, .25, .125):
            candidate = (x-delta*scale).clamp(0, 1)
            candidate_error = (trilinear(forward, candidate)-target).square().sum(dim=1, keepdim=True)
            better = candidate_error < best_error
            best_x = torch.where(better, candidate, best_x)
            best_error = torch.minimum(best_error, candidate_error)
        x = best_x
    error = (trilinear(forward, x)-target).abs().amax(dim=1)
    inverse = x.reshape(3, dim, dim, dim).contiguous()
    diagnostics = {'inverse_dim': dim, 'steps': steps,
                   'grid_p95_max_channel_error_codes': float(torch.quantile(error.flatten(), .95)*255),
                   'grid_max_channel_error_codes': float(error.max()*255),
                   'grid_residual_fraction_over_2_codes': float((error > 2/255).float().mean())}
    return inverse, diagnostics


@torch.inference_mode()
def apply_inverse(inverse, forward, rgb):
    converted = _sample_inverse(inverse, rgb)
    roundtrip = trilinear(forward, converted)
    return converted, (roundtrip-rgb).abs()


@torch.inference_mode()
def quantize_log_input(forward, continuous_log, original_rgb):
    """Return the exact uint8 model input and its actual forward roundtrip error."""
    if (not torch.isfinite(continuous_log).all()
            or bool(((continuous_log < 0) | (continuous_log > 1)).any())):
        raise ValueError('Constructed model input must be finite within [0,1]')
    codes = continuous_log.mul(255).round().byte()
    roundtrip = trilinear(forward, codes.float()/255)
    return codes, (roundtrip-original_rgb).abs()
