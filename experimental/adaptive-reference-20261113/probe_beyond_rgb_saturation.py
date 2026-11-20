"""Check one-parameter saturation fitting on the pinned paired-reference experiment."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from controlled_styles import stylize
from probe_beyond_rgb_edit import verified
from probe_beyond_rgb_paired import BASELINE_SHA, pinned_report, run
from probe_beyond_rgb_transfer import write_json


LUMA = np.array([.299, .587, .114])


def fit_saturation(source, edited):
    if source.shape != edited.shape or source.dtype != np.uint8 or edited.dtype != np.uint8:
        return dict(accepted=False, reason='Mismatched RGB8 inputs')
    if np.array_equal(source, edited):
        return dict(accepted=True, gain=1., matrix=np.vstack([np.eye(3), np.zeros(3)]).tolist())

    height, width = source.shape[:2]
    scale = min(1, 256 / max(height, width))
    size = (round(width * scale), round(height * scale))
    a, b = (np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), dtype=float)
            for rgb in (source, edited))
    light_a, light_b = a @ LUMA, b @ LUMA
    gradient_a, gradient_b = (np.concatenate([np.diff(y, axis=axis).ravel() for axis in (0, 1)])
                              for y in (light_a, light_b))
    if min(gradient_a.std(), gradient_b.std()) < .005 * 255:
        return dict(accepted=False, reason='Insufficient spatial structure')
    correlation = float(np.corrcoef(gradient_a, gradient_b)[0, 1])
    if correlation < .95:
        return dict(accepted=False, reason='Spatial structure differs', gradient_correlation=correlation)

    yy, xx = np.indices(light_a.shape)
    train = (xx // 8 + yy // 8) % 2 == 0
    chroma_a, chroma_b = a - light_a[..., None], b - light_b[..., None]
    usable = train & (np.linalg.norm(chroma_a, axis=-1) >= 4)
    if int(usable.sum()) < 128:
        return dict(accepted=False, reason='Insufficient chroma support')
    basis = chroma_a[usable]
    gain = float(np.sum(basis * chroma_b[usable]) / np.sum(basis * basis))
    error = np.abs(np.clip(light_a[..., None] + gain * chroma_a, 0, 255) - b)[~train]
    diagnostics = dict(gain=gain, gradient_correlation=correlation, train_pixels=int(usable.sum()),
                       holdout_mae=float(error.mean()), holdout_p95=float(np.quantile(error, .95)),
                       holdout_luma_mae=float(np.abs(light_a[~train] - light_b[~train]).mean()))
    if not (0 <= gain <= 2 and diagnostics['holdout_mae'] <= 1.5 and diagnostics['holdout_p95'] <= 4
            and diagnostics['holdout_luma_mae'] <= .6):
        return dict(diagnostics, accepted=False, reason='Not a certified saturation-only edit')
    matrix = gain * np.eye(3) + (1 - gain) * LUMA[:, None]
    return dict(diagnostics, accepted=True, matrix=np.vstack([matrix, np.zeros(3)]).tolist())


def contrast_controls(prepared, previous, out):
    baseline = pinned_report(previous, BASELINE_SHA)
    checks = {}
    for key, spec in baseline['inputs'].items():
        original = verified(prepared / f'{key}.png', spec['image_sha256'])
        contrast = stylize(original, np.zeros(original.shape[:2], dtype=bool), 'global_contrast')
        checks[key] = fit_saturation(original, contrast)
    write_json(out / 'contrast-controls.json', checks)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('prepared', 'previous', 'edited', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(**vars(args), fit_reference=fit_saturation)
    contrast_controls(args.prepared, args.previous, args.out)
