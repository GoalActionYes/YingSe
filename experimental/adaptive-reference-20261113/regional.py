"""Independent regional residual candidate; not a native or semantic engine.

Masks are externally supplied matching class probabilities, HxWxK, sum <= 1.
Statistics alone confound content and style. Do not enable by default without
natural-image evaluation. All edits operate on the already-global-graded image.
"""
import numpy as np

VERSION = 'regional-diagonal-residual-probe-v1'


def _validate(rgb, masks):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError('Expected nonempty RGB8')
    m = np.asarray(masks, dtype=np.float32)
    if (m.ndim != 3 or m.shape[:2] != rgb.shape[:2] or m.shape[2] < 1 or
            not np.isfinite(m).all() or np.any(m < 0) or np.any(m > 1) or
            np.any(m.sum(axis=2) > 1.0001)):
        raise ValueError('Expected matching finite class probabilities summing to <=1')
    # Ambiguous edges/classes receive no extra processing; no hard argmax.
    return np.clip((m - .8) / .2, 0, 1)


def _stats(rgb, weights):
    support = float(weights.sum())
    if support < max(32, weights.size * .01):
        return None
    x = rgb.astype(np.float64)
    mean = (x * weights[..., None]).sum(axis=(0, 1)) / support
    variance = ((x - mean)**2 * weights[..., None]).sum(axis=(0, 1)) / support
    return mean, np.sqrt(variance), support / weights.size


def fit(global_rgb, reference_rgb, target_masks, reference_masks):
    a = _validate(global_rgb, target_masks)
    b = _validate(reference_rgb, reference_masks)
    if a.shape[2] != b.shape[2]:
        raise ValueError('Region class definitions must match')
    regions = []
    for k in range(a.shape[2]):
        source = _stats(global_rgb, a[..., k])
        reference = _stats(reference_rgb, b[..., k])
        if source is None or reference is None:
            regions.append(dict(active=False, reason='Insufficient reliable class support'))
            continue
        mean, std, fraction = source
        refmean, refstd, reffraction = reference
        # Near-flat distributions cannot justify contrast amplification.
        gain = np.ones(3)
        usable = (std >= 5) & (refstd >= 5)
        gain[usable] = np.clip(refstd[usable] / std[usable], .75, 4/3)
        offset = np.clip(refmean - mean, -24, 24)
        regions.append(dict(active=True, mean=mean.tolist(), gain=gain.tolist(),
                            offset=offset.tolist(), source_fraction=fraction,
                            reference_fraction=reffraction))
    active = [r for r in regions if r['active']]
    if len(active) < 2:
        for region in regions:
            region.update(active=False, reason='Fewer than two matched reliable regions; preserve global path')
    return dict(version=VERSION, regions=regions, maximum_residual=32,
                limitations='Matching labels do not establish matching material or lighting style')


def apply(global_rgb, masks, model, strength=1.):
    weights = _validate(global_rgb, masks)
    if (isinstance(strength, (bool, np.bool_)) or not np.isscalar(strength) or
            not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Expected finite strength in [0,1]')
    if model.get('version') != VERSION or len(model['regions']) != weights.shape[2]:
        raise ValueError('Incompatible regional model')
    result = np.empty_like(global_rgb)
    for start in range(0, global_rgb.shape[0], 64):
        x = global_rgb[start:start+64].astype(np.float32)
        residual = np.zeros_like(x)
        for k, region in enumerate(model['regions']):
            if not region['active']:
                continue
            mean, gain, offset = (np.asarray(region[key], dtype=np.float32)
                                  for key in ('mean', 'gain', 'offset'))
            if any(z.shape != (3,) or not np.isfinite(z).all() for z in (mean, gain, offset)):
                raise ValueError('Invalid regional transform')
            delta = np.clip((x - mean) * (gain - 1) + offset, -32, 32)
            residual += weights[start:start+64, :, k, None] * delta
        result[start:start+64] = np.rint(np.clip(x + float(strength) * residual, 0, 255)).astype(np.uint8)
    return result
