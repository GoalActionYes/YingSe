"""Test target-conditional weights on the existing reference chroma histograms."""
import argparse
from pathlib import Path

import torch

from probe_reference_chroma_spread import main


def cells(ycbcr):
    bands = (ycbcr[0, 0] * 4).floor().clamp(0, 3).long()
    cb, cr = ycbcr[0, 1:]
    hues = (((torch.atan2(cr, cb) / (2 * torch.pi)) % 1) * 8).floor().long().clamp(0, 7)
    chromatic = torch.hypot(cb, cr) >= .04
    return bands, hues, chromatic


def weight_chroma(reference, original, target):
    if torch.equal(reference, target):
        return original.clone()
    target_bands, target_hues, colored = cells(target)
    target_cells = 8 * target_bands + target_hues
    chromatic = torch.bincount(target_cells[colored], minlength=32).reshape(4, 8)
    total = torch.bincount(target_bands.flatten(), minlength=4)
    neighbors = chromatic + chromatic.roll(1, dims=1) + chromatic.roll(-1, dims=1)
    support = neighbors / total[:, None].clamp_min(1)

    ref_bands, ref_hues, ref_colored = cells(reference)
    relative = support[ref_bands, ref_hues].div(.02).clamp(max=1)
    weights = torch.where(ref_colored, .5 + .5 * relative, 1.).flatten()
    bins = ((reference[0, 1:] + .5) * 128).floor().clamp(0, 127).long().flatten(1)
    # Use deterministic exact count order on CPU instead of GPU atomic float adds.
    counts = torch.zeros((2, 128), dtype=torch.float64)
    raw = counts.scatter_add(1, bins.cpu(), torch.ones_like(bins, device='cpu', dtype=counts.dtype))
    torch.testing.assert_close(raw / bins.shape[1], original[0, 1:].cpu().double(), atol=1e-7, rtol=0)
    weighted = counts.scatter_add(1, bins.cpu(), weights.cpu().double().expand(2, -1))
    result = original.clone()
    result[0, 1:] = (weighted / weighted.sum(dim=1, keepdim=True)).to(original)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    options = parser.parse_args()
    script = Path(__file__)
    main(options.out, script.with_name('target-hue-support-protocol.md'), weight_chroma, script)
