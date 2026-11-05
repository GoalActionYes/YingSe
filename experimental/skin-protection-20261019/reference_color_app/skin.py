"""Optional, offline semantic skin protection, independent of the recovered LUT math.

No MediaPipe/OpenCV import or model IO occurs until protection is requested.
One inference engine is serialized; temporal history belongs to each SkinSession.
"""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading
import numpy as np

VERSION = 'skin-chroma075-v1'
MODEL_SHA256 = 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'selfie_multiclass_256x256.tflite'
MAX_EDGE = 512
CACHE_LIMIT = 4
_engine_lock = threading.RLock()
_engine = None
_cv = None
_mp = None
_mask_cache = OrderedDict()


class SkinProtectionError(ValueError):
    """An explicit protection failure; callers must not report success."""


def _get_engine():
    global _engine, _cv, _mp
    # Caller holds the inference lock, including model initialization.
    if _engine is not None:
        return _engine
    try:
        cache = Path(__file__).resolve().parents[1] / 'data' / 'mpl-cache'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(cache)
        os.environ['MPLBACKEND'] = 'Agg'
        import cv2
        import mediapipe as mp
        model = MODEL_PATH.read_bytes()
        if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
            raise SkinProtectionError('肤色保护模型校验失败，请重新解压完整实验版。')
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model),
            output_category_mask=True, output_confidence_masks=True)
        engine = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        _cv, _mp, _engine = cv2, mp, engine
        return engine
    except SkinProtectionError:
        raise
    except Exception as exc:
        raise SkinProtectionError('肤色保护模型无法加载，请检查完整模型和依赖，或关闭保护后继续。') from exc


def _validate(original, graded):
    for value in (original, graded):
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or
                value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1):
            raise SkinProtectionError('肤色保护只接受有效的8位RGB画面。')
    if original.shape != graded.shape:
        raise SkinProtectionError('肤色保护原图与调色画面尺寸不一致。')


def _predict(original):
    """Return a copied <=512 mask/gray and immutable scalar diagnostics."""
    h, w = original.shape[:2]
    # Full target RGB identity deliberately excludes the reference/graded image.
    digest = hashlib.sha256()
    digest.update(f'{VERSION}:{MODEL_SHA256}:{h}:{w}'.encode('ascii'))
    for row in range(0, h, 64):
        digest.update(np.ascontiguousarray(original[row:row + 64]).tobytes())
    key = digest.hexdigest()
    with _engine_lock:
        cached = _mask_cache.get(key)
        if cached is not None:
            _mask_cache.move_to_end(key)
            mask, gray, diag = cached
            return mask.copy(), gray.copy(), dict(diag, cache_hit=True)
        engine = _get_engine()
        scale = min(1.0, MAX_EDGE / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        small = _cv.resize(original, size, interpolation=_cv.INTER_AREA) if scale < 1 else original
        small = np.ascontiguousarray(small)
        try:
            result = engine.segment(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=small))
            category = np.asarray(result.category_mask.numpy_view()).squeeze().copy()
            confidence = (np.asarray(result.confidence_masks[2].numpy_view()).squeeze().copy() +
                          np.asarray(result.confidence_masks[3].numpy_view()).squeeze())
            if category.shape != small.shape[:2] or confidence.shape != category.shape or not np.isfinite(confidence).all():
                raise ValueError('invalid model mask')
            ramp = np.clip((confidence - .45) / .40, 0, 1)
            mask = (ramp * ramp * (3 - 2 * ramp) * np.isin(category, (2, 3))).astype(np.float32)
            reliable = int(np.count_nonzero(mask > .5)) >= max(8, int(np.ceil(mask.size * .001)))
            if reliable:
                mask = _cv.GaussianBlur(mask, (0, 0), 1.2)
            else:
                mask.fill(0)
            gray = _cv.cvtColor(small, _cv.COLOR_RGB2GRAY)
            diag = dict(skin_detected=bool(reliable), skin_fraction=float((mask > .5).mean()),
                        analysis_width=size[0], analysis_height=size[1], cache_hit=False,
                        model_sha256=MODEL_SHA256, protection_version=VERSION,
                        chroma_restore=.75, skin_classes=[2, 3])
        except Exception as exc:
            raise SkinProtectionError('肤色识别失败，本次未完成保护；可关闭保护后继续。') from exc
        _mask_cache[key] = (mask.copy(), gray.copy(), dict(diag))
        while len(_mask_cache) > CACHE_LIMIT:
            _mask_cache.popitem(last=False)
        return mask, gray, diag


def _blend(original, graded, mask):
    """Only one full-resolution float mask; RGB arithmetic uses bounded row chunks."""
    if not np.any(mask):
        return graded.copy()
    height, width = original.shape[:2]
    full = _cv.resize(mask, (width, height), interpolation=_cv.INTER_LINEAR)
    output = np.empty_like(graded)
    for start in range(0, height, 64):
        end = min(start + 64, height)
        base = graded[start:end].astype(np.float32)
        delta = original[start:end].astype(np.float32) - base
        luma = delta @ np.array([.299, .587, .114], np.float32)
        delta -= luma[..., None]
        base += .75 * full[start:end, :, None] * delta
        output[start:end] = np.rint(np.clip(base, 0, 255)).astype(np.uint8)
    return output


class SkinSession:
    """Per-clip temporal history. Constructing or resetting never loads the model."""
    def __init__(self):
        self._lock = threading.RLock()
        self._previous_gray = None
        self._previous_mask = None

    def reset(self):
        with self._lock:
            self._previous_gray = None
            self._previous_mask = None

    def _temporal(self, mask, gray):
        if self._previous_gray is None or gray.shape != self._previous_gray.shape:
            return mask, 'new_history'
        before = self._previous_gray
        # Histogram and direct difference jointly reject cuts, not ordinary pans alone.
        hist_now = np.bincount((gray.ravel() // 8), minlength=32) / gray.size
        hist_old = np.bincount((before.ravel() // 8), minlength=32) / before.size
        difference = float(np.abs(gray.astype(np.float32) - before).mean() / 255)
        if difference > .30 or (difference > .12 and np.abs(hist_now - hist_old).sum() > .65):
            return mask, 'scene_cut'
        try:
            flow = _cv.calcOpticalFlowFarneback(gray, before, None, .5, 3, 15, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]].astype(np.float32)
            map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
            previous = _cv.remap(self._previous_mask, map_x, map_y, _cv.INTER_LINEAR, borderMode=_cv.BORDER_CONSTANT)
            old_gray = _cv.remap(before, map_x, map_y, _cv.INTER_LINEAR, borderMode=_cv.BORDER_CONSTANT)
            valid = (map_x >= 0) & (map_x < gray.shape[1] - 1) & (map_y >= 0) & (map_y < gray.shape[0] - 1)
            residual = np.abs(gray.astype(np.float32) - old_gray) / 255
            if not np.any(valid) or float(residual[valid].mean()) > .12:
                return mask, 'untrusted_motion'
            # Current evidence gates history; an old face cannot survive on a missing face.
            trust = valid & (residual < .12) & (mask > .05)
            stable = np.where(trust, .75 * mask + .25 * previous, mask).astype(np.float32)
            return stable, 'motion_aligned'
        except Exception:
            # Motion failure uses the current successful semantic result, never stale history.
            return mask, 'motion_reset'

    def process(self, original_rgb, graded_rgb, temporal=False):
        _validate(original_rgb, graded_rgb)
        if not isinstance(temporal, bool):
            raise SkinProtectionError('肤色保护时间模式必须为布尔值。')
        with self._lock:
            try:
                mask, gray, diag = _predict(original_rgb)
                if not diag['skin_detected']:
                    self.reset()
                    return graded_rgb.copy(), dict(diag, temporal_state='no_skin', message='未检测到可靠皮肤，保留整体调色结果。')
                state = 'static'
                if temporal:
                    mask, state = self._temporal(mask, gray)
                    self._previous_gray, self._previous_mask = gray.copy(), mask.copy()
                else:
                    self.reset()
                return _blend(original_rgb, graded_rgb, mask), dict(diag, temporal_state=state, message='已应用人物肤色保护。')
            except SkinProtectionError:
                self.reset()
                raise
            except Exception as exc:
                self.reset()
                raise SkinProtectionError('肤色保护处理失败，本次未完成保护；可关闭保护后继续。') from exc


def protect_image(original_rgb, graded_rgb):
    """Static preview/PNG entry, no shared video history; inference remains serialized."""
    return SkinSession().process(original_rgb, graded_rgb, temporal=False)
