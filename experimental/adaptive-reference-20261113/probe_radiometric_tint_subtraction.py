"""Subtract chroma from a color-ratio-preserving virtual reference."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_bg_real_channel_replay import histogram, infer
from probe_dast_exposure_channels import fragments
from probe_dast_radiometric_control import ANALYSIS_SIZE, radiometric_match
from probe_luma_chroma_decouple import combine_linear, linear, run

FLOOR = 1. / (255. * 12.92)


def analysis_rgb(rgb):
    return np.array(Image.fromarray(rgb).resize(ANALYSIS_SIZE, Image.Resampling.BILINEAR))


def renderer(engine, target):
    low, target_hist = fragments(engine, target)
    anchor_rgb = infer(engine, target, low, target_hist, target_hist)
    anchor_linear = linear(anchor_rgb)
    guide = analysis_rgb(target)

    def apply(reference, baseline):
        full = infer(engine, target, low, target_hist, histogram(engine, reference))
        if not np.array_equal(full, baseline):
            raise ValueError("Full configuration failed pixel replay")
        virtual, error, clipped = radiometric_match(guide, analysis_rgb(reference))
        virtual_rgb = infer(engine, target, low, target_hist, histogram(engine, virtual))
        corrected = linear(full) * ((anchor_linear + FLOOR) / (linear(virtual_rgb) + FLOOR))
        output, diagnostics = combine_linear(full, corrected)
        return output, {**diagnostics, "virtual_luma_mae": error, "virtual_unattainable_fraction": clipped}

    return apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "replay", "lut", "editor", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.prepared, args.replay, args.lut, args.editor, args.output, renderer_factory=renderer)
