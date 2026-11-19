"""Research candidate that briefly holds a grid during localized disruptions."""

import numpy as np
from PIL import Image
import torch

from spatial_render import PreparedSpatial, validate_rgb
from temporal_grid import TemporalGrid


class OcclusionHold(TemporalGrid):
    def __init__(self):
        super().__init__(relative_cut=True)

    def reset(self):
        super().reset()
        self.held = 0

    @torch.inference_mode()
    def smooth(self, prepared, frame):
        validate_rgb(frame)
        if (self.previous is not None and self.engine is prepared.engine
                and self.dimensions == frame.shape and self.held < 15):
            thumbnail = np.array(Image.fromarray(frame).resize((32, 32), Image.Resampling.BILINEAR),
                                 dtype=np.float32) / 255
            tiles = np.abs(thumbnail - self.thumbnail).reshape(4, 8, 4, 8, 3).mean(axis=(1, 3, 4))
            if (tiles <= .035).sum() >= 7 and (tiles >= .10).sum() >= 4:
                self.held += 1
                diagnostics = dict(prepared.diagnostics, temporal_reset_reason=None,
                                   temporal_occlusion_hold=True)
                return PreparedSpatial(prepared.engine, self.previous.clone(), diagnostics)
        self.held = 0
        return super().smooth(prepared, frame)
