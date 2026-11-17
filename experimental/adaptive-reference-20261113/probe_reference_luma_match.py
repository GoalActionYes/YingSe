"""Retain reference luminance and use target luminance to weight reference chroma."""
from pathlib import Path

import numpy as np
import torch

from probe_reference_chroma_spread import BASE, main


def luma_bins(ycbcr):
    return (ycbcr[0, 0].flatten() * 8).floor().clamp(0, 7).long().cpu().numpy()


def luma_matched_histogram(reference, original, target):
    reference_bins = luma_bins(reference)
    reference_mass = np.bincount(reference_bins, minlength=8).astype(np.float64)
    reference_mass /= len(reference_bins)
    target_bins = luma_bins(target)
    target_mass = np.bincount(target_bins, minlength=8).astype(np.float64)
    target_mass /= len(target_bins)
    supported = reference_mass > 0
    target_mass = target_mass * supported + target_mass[~supported].sum() * reference_mass
    desired_mass = (reference_mass + target_mass) * .5
    weights = desired_mass[reference_bins] / reference_mass[reference_bins]
    chroma = ((reference[0, 1:].flatten(1) + .5) * 128).floor().clamp(0, 127).long().cpu().numpy()
    counts = np.array([np.bincount(channel, minlength=128) for channel in chroma])
    np.testing.assert_allclose(counts / len(reference_bins), original[0, 1:].cpu().numpy(), atol=1e-7, rtol=0)
    weighted = np.array([np.bincount(channel, weights=weights, minlength=128) for channel in chroma])
    weighted /= weighted.sum(axis=1, keepdims=True)
    result = original.clone()
    result[0, 1:] = torch.as_tensor(weighted, dtype=original.dtype, device=original.device)
    return result


if __name__ == '__main__':
    main(BASE / 'reference-luma-match-v1', Path(__file__).with_name('reference-luma-match-protocol.md'),
         luma_matched_histogram, __file__)
