"""Inference-only Torch equivalents of SA-LUT's CUDA forward operations.

Author reference: Ry3nG/SA-LUT, commit 3e62f9c9b59ea7b30d220ff6ba61047dab898c9a,
quadrilinear_cpp/src/quadrilinear4d_kernel.cu. The CPU extension uses a different
layout and is intentionally NOT the compatibility target. No training backward.
"""
import torch
import torch.nn.functional as F


def _validate_image(image, channels):
    if image.ndim != 4 or image.shape[1] != channels or not image.is_floating_point():
        raise ValueError('Expected floating NCHW image')
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError('Interpolation input must be finite and within [0,1]')


def trilinear(lut, image):
    _validate_image(image, 3)
    if lut.ndim == 4:
        lut = lut.unsqueeze(0)
    if (lut.ndim != 5 or lut.shape[1] != 3 or lut.shape[0] not in (1, image.shape[0])
            or len(set(lut.shape[-3:])) != 1 or lut.shape[-1] < 2):
        raise ValueError('Expected [N,3,D,D,D] LUT')
    if lut.device != image.device or lut.dtype != image.dtype:
        raise ValueError('LUT and image must share dtype and device')
    grid = (image.permute(0, 2, 3, 1) / 1.000001 * 2 - 1).unsqueeze(1)
    return F.grid_sample(lut.expand(image.shape[0], -1, -1, -1, -1), grid,
                         mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)


def quadrilinear(lut, image):
    _validate_image(image, 4)
    if lut.ndim == 6 and lut.shape[0] == 1:
        lut = lut[0]
    if lut.ndim != 5 or lut.shape[:2] != (3, 2):
        raise ValueError('Expected [3,2,D,D,D] LUT with exactly two context bins')
    context = image[:, :1]
    first = trilinear(lut[:, 0], image[:, 1:])
    second = trilinear(lut[:, 1], image[:, 1:])
    return first * (1-context) + second * context


def efficient_attention(q, k, v, temperature, chunk_rows=128):
    """Same softmax(QK/sqrt(d)*temperature)V, bounded score-matrix memory.

    No fp16 conversion or resolution reduction. Chunk queries only: every query
    still sees ALL keys. Unlike chunking keys, this preserves softmax semantics.
    """
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    chunks = []
    key = k.transpose(-1, -2).unsqueeze(1)
    value = v.unsqueeze(1)
    for start in range(0, q.shape[1], chunk_rows):
        query = (q[:, start:start+chunk_rows] * temperature).unsqueeze(1)
        chunks.append(F.scaled_dot_product_attention(query, key, value, dropout_p=0).squeeze(1))
    return torch.cat(chunks, dim=1)


class TrilinearInterpolation(torch.nn.Module):
    def forward(self, lut, image):
        return trilinear(lut, image)


class QuadrilinearInterpolation4D(torch.nn.Module):
    def forward(self, lut, image):
        return lut, quadrilinear(lut, image)
