"""Existing offline selfie model adapter; labels are person parts, not scene semantics."""
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'experimental/skin-protection-20261019/models/selfie_multiclass_256x256.tflite'
SHA256 = 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0'
LABELS = ['background', 'hair', 'body_skin', 'face_skin', 'clothes', 'accessories']


class Segmenter:
    def __init__(self):
        import mediapipe as mp
        model = MODEL.read_bytes()
        if hashlib.sha256(model).hexdigest() != SHA256:
            raise ValueError('Existing offline model checksum mismatch')
        self.mp = mp
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model),
            output_category_mask=False, output_confidence_masks=True)
        self.engine = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def close(self):
        self.engine.close()

    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB8')
        start = time.perf_counter()
        small = Image.fromarray(rgb)
        small.thumbnail((512, 512), Image.Resampling.BILINEAR)
        result = self.engine.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(small)))
        if len(result.confidence_masks) != len(LABELS):
            raise ValueError('Unexpected model label count')
        masks = np.stack([np.asarray(Image.fromarray(np.array(m.numpy_view(), copy=True).squeeze()).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)) for m in result.confidence_masks], axis=2)
        if not np.isfinite(masks).all():
            raise ValueError('Nonfinite segmentation')
        masks = np.clip(masks, 0, 1)
        masks /= np.maximum(1, masks.sum(axis=2, keepdims=True))
        return masks, dict(labels=LABELS, model_sha256=SHA256,
                           seconds=time.perf_counter()-start,
                           limitation='Selfie part segmentation; not general material or lighting recognition')
