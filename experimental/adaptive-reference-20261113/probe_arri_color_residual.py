"""Screen within-reference color residuals from a frozen visual encoder."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dense_anchor_model import DenseAnchorModel
from probe_arri_edge_coherence import LUMA, run, states

ARMS = ("rgb_features", "color_residual")


class ReferenceFeatures:
    def __init__(self):
        self.encoder = DenseAnchorModel()
        self.cache = {}

    def __call__(self, rgb):
        key = (rgb.shape, sha256(rgb.tobytes()).digest())
        if key not in self.cache:
            if len(self.cache) == 68:
                raise ValueError("Reference feature budget exceeded")
            photo = Image.fromarray(rgb)
            photo.thumbnail((224, 224), Image.Resampling.BILINEAR)
            small = np.asarray(photo)
            gray = np.rint(small @ LUMA).astype(np.uint8)
            mono = np.repeat(gray[..., None], 3, axis=2)
            color, colorless = (self.encoder.extract(arm)[0].mean(dim=0).cpu().numpy().astype(np.float64)
                                for arm in (small, mono))
            self.cache[key] = (color, color - colorless)
        return self.cache[key]

    def fit(self, fitting, cube):
        samples = [self(rgb) for group in fitting for image in group.values()
                   for rgb in states(image, cube).values()]
        if len(samples) != 48 or len(self.cache) != 48:
            raise ValueError("Unexpected reference fitting cohort")
        response = np.tile([-.5, .5], len(samples) // 2)
        models = {}
        for name, observations in zip(ARMS, zip(*samples, strict=True), strict=True):
            x = np.stack(observations)
            center, spread = x.mean(axis=0), x.std(axis=0)
            scale = np.where(spread > 1e-12, spread, 1)
            z = (x - center) / scale
            weights = z.T @ np.linalg.solve(z @ z.T + np.eye(len(z)), response)
            models[name] = (center, scale, weights)
        return models

    def predict(self, reference, models):
        return {arm: float((vector - center) / scale @ weights + .5)
                for vector, (arm, (center, scale, weights)) in zip(self(reference), models.items(), strict=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdr_cube", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    features = ReferenceFeatures()
    provenance = {**features.encoder.provenance, "maximum_input_edge": 224, "mean_patch_tokens": True}
    run(args.sdr_cube, args.native, args.output, fitter=features.fit, predictor=features.predict,
        arms=ARMS, provenance=provenance, post_sdr_control=False)
