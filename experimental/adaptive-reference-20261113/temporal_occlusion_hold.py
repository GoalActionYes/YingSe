"""Research candidate that briefly holds a grid during localized disruptions."""

import numpy as np
from PIL import Image
import torch

from spatial_render import PreparedSpatial, validate_rgb
from temporal_grid import TemporalGrid


def thumbnail_of(frame):
    return np.array(Image.fromarray(frame).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def tile_difference(left, right):
    return np.abs(left - right).reshape(4, 8, 4, 8, 3).mean(axis=(1, 3, 4))


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
            thumbnail = thumbnail_of(frame)
            tiles = tile_difference(thumbnail, self.thumbnail)
            if (tiles <= .035).sum() >= 7 and (tiles >= .10).sum() >= 4:
                self.held += 1
                diagnostics = dict(prepared.diagnostics, temporal_reset_reason=None,
                                   temporal_occlusion_hold=True)
                return PreparedSpatial(prepared.engine, self.previous.clone(), diagnostics)
        self.held = 0
        return super().smooth(prepared, frame)


class OcclusionEpisode(TemporalGrid):
    def __init__(self):
        super().__init__(relative_cut=True)

    def reset(self):
        super().reset()
        self.episode = None
        self.held = 0

    @torch.inference_mode()
    def smooth(self, prepared, frame):
        validate_rgb(frame)
        if self.previous is None or self.engine is not prepared.engine or self.dimensions != frame.shape:
            self.episode = None
            self.held = 0
            return super().smooth(prepared, frame)

        thumb = thumbnail_of(frame)
        if self.episode is None:
            tiles = tile_difference(thumb, self.thumbnail)
            stable, changed = tiles <= .035, tiles >= .10
            if stable.sum() < 7 or changed.sum() < 4:
                return super().smooth(prepared, frame)
            self.episode = (self.thumbnail, thumb, stable, changed)
            self.held = 0
        else:
            clear, blocked, stable, changed = self.episode
            to_clear = tile_difference(thumb, clear)
            to_blocked = tile_difference(thumb, blocked)
            recovered = (to_clear[changed] < to_blocked[changed]).mean() > .5
            scene_changed = (np.minimum(to_clear[stable], to_blocked[stable]) >= .10).mean() > .5
            if recovered or scene_changed:
                needs_reset = scene_changed or self.held >= 15
                self.episode = None
                self.held = 0
                if needs_reset:
                    self.reset()
                return super().smooth(prepared, frame)
        if self.held >= 15:
            return super().smooth(prepared, frame)

        self.held += 1
        diagnostics = dict(prepared.diagnostics, temporal_reset_reason=None, temporal_occlusion_hold=True)
        return PreparedSpatial(prepared.engine, self.previous.clone(), diagnostics)
