"""Parametric auto-match: a CANDIDATE second mode alongside the preset-blend mode.

Read this before using or citing it.

What this module is
-------------------
It estimates one candidate set of grading parameters from the target footage AND
the reference image jointly:

    estimate_params(target, reference, constraints) -> P

What it explicitly is NOT
-------------------------
* It does NOT recover the parameters the reference image's colourist actually
  used. Reference statistics conflate scene content with grade: an ungraded snow
  scene is already bright; a lifted night shot can still be dark. The output is
  a candidate, not ground truth.
* It is NOT claimed to be better than the preset-blend mode. Whether it is
  depends on measurements in compare_modes.py, which have not been run yet.

Design constraints applied on purpose
-------------------------------------
* Parameter count and range are deliberately limited (see PARAM_SPEC). Overlapping
  controls (a free tone curve plus gamma plus lift plus gain) would create fitting
  ambiguity, so tone is modelled with three de-coupled scalars only.
* Shadow / highlight tint are projected onto the zero-luminance plane before use,
  which removes their coupling with the exposure and contrast parameters.
* The uniform chroma component and the luminance-dependent chroma component are
  estimated separately, so white balance and split toning do not fight over the
  same residual. Residual coupling remains and is documented.

Rendering order follows a conventional primary-grade chain and puts saturation
last, so it acts on the split-tone dye rather than before it.
"""

import numpy as np

LUMA_W = np.array([.2126, .7152, .0722], dtype=np.float32)
PIVOT = 0.5

# name, low, high, neutral. 11 stored values; shadow/highlight tint are projected
# onto the zero-luminance plane, so each contributes 2 effective degrees of
# freedom: 5 scalars + 2 + 2 = 9 effective.
PARAM_SPEC = [
    ('exposure',    -1.50,  1.50,  0.0),
    ('contrast',     0.60,  1.80,  1.0),
    ('scurve',      -0.24,  0.24,  0.0),   # bound keeps the tone response monotone
    ('saturation',   0.40,  1.80,  1.0),
    ('temp',        -1.00,  1.00,  0.0),
    ('shadow_r',    -0.06,  0.06,  0.0),
    ('shadow_g',    -0.06,  0.06,  0.0),
    ('shadow_b',    -0.06,  0.06,  0.0),
    ('high_r',      -0.06,  0.06,  0.0),
    ('high_g',      -0.06,  0.06,  0.0),
    ('high_b',      -0.06,  0.06,  0.0),
]
N_PARAM = len(PARAM_SPEC)
_NAMES = [s[0] for s in PARAM_SPEC]
_LOW = np.array([s[1] for s in PARAM_SPEC], dtype=np.float64)
_HIGH = np.array([s[2] for s in PARAM_SPEC], dtype=np.float64)
_NEUTRAL = np.array([s[3] for s in PARAM_SPEC], dtype=np.float64)


def neutral():
    return _NEUTRAL.copy()


def clip_params(p):
    return np.clip(np.asarray(p, dtype=np.float64), _LOW, _HIGH)


def params_dict(p):
    return {k: float(v) for k, v in zip(_NAMES, np.asarray(p, dtype=np.float64))}


def luminance(x):
    return (x * LUMA_W).sum(-1, keepdims=True)


def _zero_luma(o):
    """Project an RGB offset onto the plane that leaves luminance unchanged."""
    o = np.asarray(o, dtype=np.float32)
    return o - (o * LUMA_W).sum(-1, keepdims=True)


def apply_params(rgb, p, blend=1.0):
    """Pointwise pipeline: exposure -> contrast -> S -> white balance -> split -> sat.

    Every step is pointwise, so the whole thing bakes into one 3D LUT and is
    reused for every frame. That keeps the structural properties the existing
    pipeline already relies on; it does not by itself make the result better.
    """
    x = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    src = x
    p = np.asarray(p, dtype=np.float32)
    exposure, contrast, scurve, sat, temp = p[0], p[1], p[2], p[3], p[4]
    shadow, high = p[5:8], p[8:11]

    # --- tone: linear part -------------------------------------------------
    x = x * (2.0 ** exposure)
    x = (x - PIVOT) * contrast + PIVOT
    x = np.clip(x, 0, 1)
    # --- tone: symmetric S deviation, zero at 0 / .5 / 1 -------------------
    if abs(float(scurve)) > 1e-6:
        x = np.clip(x + scurve * 4.0 * x * (1 - x) * (2 * x - 1), 0, 1)
    # --- white balance: channel gains, luminance preserved -----------------
    g = np.array([1 + .25 * temp, 1.0, 1 - .25 * temp], dtype=np.float32)
    g = g / (g * LUMA_W).sum()
    x = np.clip(x * g, 0, 1)
    # --- split tone: additive dye weighted by luminance --------------------
    y = luminance(x)
    sh = _zero_luma(shadow)
    hi = _zero_luma(high)
    x = np.clip(x + (1 - y) * sh + y * hi, 0, 1)
    # --- saturation last, so it acts on the dye ----------------------------
    y = luminance(x)
    x = np.clip(y + (x - y) * sat, 0, 1)
    # --- strength ----------------------------------------------------------
    if float(blend) < 1.0:
        x = np.clip((1 - float(blend)) * src + float(blend) * x, 0, 1)
    return x.astype(np.float32)


def params_to_lut(p, size=33, blend=1.0):
    if size not in (17, 33, 65):
        raise ValueError('LUT size must be 17, 33 or 65')
    b, g, r = np.meshgrid(*([np.linspace(0, 1, size, dtype=np.float32)] * 3), indexing='ij')
    return apply_params(np.stack([r, g, b], -1), p, blend)


def apply_lut(rgb, lut):
    """Trilinear interpolation; LUT indexed [blue, green, red, channel]."""
    p = np.clip(rgb, 0, 1) * (len(lut) - 1)
    lo = np.floor(p).astype(np.int32)
    hi = np.minimum(lo + 1, len(lut) - 1)
    f = p - lo
    out = np.zeros_like(rgb, dtype=np.float32)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                rr = hi[..., 0] if dr else lo[..., 0]
                gg = hi[..., 1] if dg else lo[..., 1]
                bb = hi[..., 2] if db else lo[..., 2]
                factor = ((f[..., 0] if dr else 1 - f[..., 0])
                          * (f[..., 1] if dg else 1 - f[..., 1])
                          * (f[..., 2] if db else 1 - f[..., 2]))
                out += lut[bb, gg, rr] * factor[..., None]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)

# Per-iteration caps on the chroma stage. The first version let the temperature
# step compound unclamped and hit its bound within three iterations, which is what
# produced the blue cast. These caps are a guard against that, not a fix for the
# underlying reasoning.
_MAX_TEMP_STEP = 0.15
_MAX_TINT_STEP = 0.02
_TEMP_GAIN = 0.6


def _downsample(x, side=160):
    h, w = x.shape[:2]
    step = max(1, int(round(max(h, w) / side)))
    return np.ascontiguousarray(x[::step, ::step])


def _resid(rgb):
    """Chroma residual (R-Y, G-Y, B-Y). Rank 2 by construction."""
    return rgb - luminance(rgb)


def _bin_index(y, n):
    return np.clip((y * n).astype(np.int32), 0, n - 1)


def _bin_resid(rgb, n=10, min_frac=0.01):
    """Mean chroma residual per fixed luminance bin; thin bins are dropped."""
    y = luminance(rgb).ravel()
    idx = _bin_index(y, n).ravel()
    res = _resid(rgb).reshape(-1, 3)
    total = len(idx)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for b in range(n):
        m = idx == b
        if m.sum() >= min_frac * total:
            out[b] = res[m].mean(0)
    return out


def _fit_tone(cur_lum, ref_lum):
    """Least-squares fit of (exposure, contrast) mapping target quantiles to reference.

    Model:  out = (x * 2^e - pivot) * c + pivot
    Writing a = 2^e and b = 1/c gives a linear system in (a, b):
        x * a - (r - pivot) * b = pivot
    solved over nine quantile pairs. Nine equations, two unknowns, so the fit is
    over-determined rather than exact, which is what we want from noisy statistics.
    """
    tq = np.percentile(cur_lum, _QS)
    rq = np.percentile(ref_lum, _QS)
    A = np.stack([tq, -(rq - PIVOT)], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.full(len(_QS), PIVOT), rcond=None)
    a, b = float(sol[0]), float(sol[1])
    a = min(max(a, 1e-3), 1e3)
    b = min(max(b, 1e-3), 1e3)
    return np.log2(a), 1.0 / b


def estimate_params(target, reference, constraints=None, iterations=3,
                    damping=0.7, bins=10, side=160):
    """Joint estimation from target AND reference.

    Every stage compares the CURRENT rendering of the target against the
    reference at matched luminance, then accumulates a corrective step. Nothing
    is read off the reference alone.
    """
    constraints = {} if constraints is None else constraints
    damp = float(constraints.get('damping', damping))
    n = int(constraints.get('bins', bins))

    tgt = _downsample(np.clip(np.asarray(target, dtype=np.float32), 0, 1), side)
    ref = _downsample(np.clip(np.asarray(reference, dtype=np.float32), 0, 1), side)

    p = neutral()
    ref_lum = luminance(ref).ravel()
    ref_res = _bin_resid(ref, n)

    for _ in range(int(iterations)):
        cur = apply_params(tgt, p)
        cur_lum = luminance(cur).ravel()
        cur_res = _bin_resid(cur, n)

        # --- tone ----------------------------------------------------------
        e_step, c_step = _fit_tone(cur_lum, ref_lum)
        p[0] += damp * e_step
        if c_step > 0:
            p[1] *= (c_step ** damp)

        # --- chroma, at matched luminance bins -----------------------------
        valid = ~np.isnan(ref_res[:, 0]) & ~np.isnan(cur_res[:, 0])
        if valid.sum() >= 3:
            d = (ref_res[valid] - cur_res[valid])          # (k, 3) additive delta
            mid = np.nanmedian(d, axis=0)                  # uniform component
            low = np.nanmedian(d[:max(2, len(d) // 5)], axis=0)
            high = np.nanmedian(d[-max(2, len(d) // 5):], axis=0)

            # uniform part drives white balance (blue minus red axis)
            p[4] += damp * float(mid[2] - mid[0]) / 0.25

            # luminance-dependent part drives split tone, relative to the uniform part
            p[5:8] += damp * (low - mid)
            p[8:11] += damp * (high - mid)

            # saturation: ratio of chroma magnitude at matched luminance
            cm = np.linalg.norm(cur_res[valid], axis=1)
            rm = np.linalg.norm(ref_res[valid], axis=1)
            keep = cm > 1e-4
            if keep.sum() >= 3:
                ratio = float(np.nanmedian(rm[keep] / cm[keep]))
                ratio = min(max(ratio, 0.4), 2.5)
                p[3] *= (ratio ** damp)

        p = clip_params(p)

    return p, {'iterations': int(iterations), 'damping': damp, 'bins': n,
               'note': 'candidate parameters from joint target/reference statistics; '
                       'not a recovery of the reference grade'}


# ---------------------------------------------------------------------------
# v2: quantile bins and a three-parameter tone fit
# ---------------------------------------------------------------------------

def _bin_resid_q(rgb, n=10, min_frac=0.01):
    """Equal-count luminance bins.

    The first version used fixed-width bins. Once the tone mapping changes, a
    fixed bin holds a different pixel population in the reference than in the
    target, so bin-wise statistics compare unrelated pixels. Equal-count bins keep
    the same pixel fraction in every bin, which makes the comparison meaningful
    across images with different tone curves.
    """
    x = np.asarray(rgb, dtype=np.float32)
    y = luminance(x).ravel()
    edges = np.unique(np.percentile(y, np.linspace(0, 100, n + 1)))
    if len(edges) < 4:
        return None
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2)
    res = (x - luminance(x)).reshape(-1, 3)
    k = len(edges) - 1
    out = np.full((k, 3), np.nan, dtype=np.float64)
    for b in range(k):
        m = idx == b
        if m.sum() >= min_frac * len(idx):
            out[b] = res[m].mean(0)
    return out


def _fit_tone3(cur_lum, ref_lum):
    """Fit (exposure, contrast, scurve) against quantile pairs.

    A linear tone model cannot lift the blacks and pull down the highlights at the
    same time, which is what a faded reference does. The S parameter is searched on
    a small grid; exposure and contrast are solved linearly for each candidate.
    """
    tq = np.percentile(cur_lum, _QS)
    rq = np.percentile(ref_lum, _QS)
    best = None
    for s in np.linspace(-0.24, 0.24, 25):
        x = np.clip(tq, 0, 1)
        xs = np.clip(x + s * 4 * x * (1 - x) * (2 * x - 1), 0, 1)
        a_mat = np.stack([xs, -(rq - PIVOT)], axis=1)
        sol, *_ = np.linalg.lstsq(a_mat, np.full(len(_QS), PIVOT), rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if not (1e-3 < a < 1e3 and 1e-3 < b < 1e3):
            continue
        pred = np.clip((xs * a - PIVOT) * (1 / b) + PIVOT, 0, 1)
        err = float(np.abs(pred - rq).mean())
        if best is None or err < best[0]:
            best = (err, float(np.log2(a)), float(1.0 / b), float(s))
    if best is None:
        return 0.0, 1.0, 0.0
    return best[1], best[2], best[3]


def estimate_params_v2(target, reference, constraints=None, tone_iters=3,
                       chroma_iters=4, damping=0.6, bins=10, side=160):
    """Same idea as estimate_params, with two defects addressed:

    1. equal-count luminance bins instead of fixed-width bins, so bin-wise chroma
       comparison is not comparing different pixel populations (see _bin_resid_q);
    2. exposure / contrast / S fitted jointly, so a reference that both lifts the
       blacks and pulls down the highlights is reachable (see _fit_tone3);
    3. chroma steps capped per iteration instead of driving temperature into its
       bound.

    Still produces a candidate, not a recovery. Still not established as better
    than the preset blend - run compare_modes.py.
    """
    constraints = {} if constraints is None else constraints
    damp = float(constraints.get('damping', damping))
    n = int(constraints.get('bins', bins))

    tgt = _downsample(np.clip(np.asarray(target, dtype=np.float32), 0, 1), side)
    ref = _downsample(np.clip(np.asarray(reference, dtype=np.float32), 0, 1), side)

    p = neutral()
    ref_lum = luminance(ref).ravel()

    # --- stage 1: tone only, to convergence --------------------------------
    for _ in range(int(tone_iters)):
        cur_lum = luminance(apply_params(tgt, p)).ravel()
        e, c, s = _fit_tone3(cur_lum, ref_lum)
        p[0] += damp * e
        if c > 0:
            p[1] *= (c ** damp)
        p[2] += damp * (s - p[2])
        p = clip_params(p)

    # --- stage 2: chroma, with the tone curve now settled ------------------
    for _ in range(int(chroma_iters)):
        cur = apply_params(tgt, p)
        cr = _bin_resid_q(cur, n)
        rr = _bin_resid_q(ref, n)
        if cr is None or rr is None:
            break
        k = min(len(cr), len(rr))
        cr, rr = cr[:k], rr[:k]
        valid = ~np.isnan(cr[:, 0]) & ~np.isnan(rr[:, 0])
        if valid.sum() < 3:
            break
        d = rr[valid] - cr[valid]
        idx = np.flatnonzero(valid)
        lo_sel = idx < max(2, len(cr) // 5)
        hi_sel = idx >= len(cr) - max(2, len(cr) // 5)
        mid = np.nanmedian(d, axis=0)
        low = np.nanmedian(d[lo_sel], axis=0) if lo_sel.any() else mid
        high = np.nanmedian(d[hi_sel], axis=0) if hi_sel.any() else mid

        step_t = np.clip(damp * float(mid[2] - mid[0]) / _TEMP_GAIN,
                         -_MAX_TEMP_STEP, _MAX_TEMP_STEP)
        p[4] += step_t
        p[5:8] += np.clip(damp * (low - mid), -_MAX_TINT_STEP, _MAX_TINT_STEP)
        p[8:11] += np.clip(damp * (high - mid), -_MAX_TINT_STEP, _MAX_TINT_STEP)

        cm_ = np.linalg.norm(cr[valid], axis=1)
        rm_ = np.linalg.norm(rr[valid], axis=1)
        keep = cm_ > 1e-4
        if keep.sum() >= 3:
            ratio = float(np.nanmedian(rm_[keep] / cm_[keep]))
            ratio = min(max(ratio, 0.5), 2.0)
            p[3] = p[3] * (ratio ** (damp * 0.6))
        p = clip_params(p)

    return p, {'estimator': 'v2-quantile-bins', 'tone_iters': int(tone_iters),
               'chroma_iters': int(chroma_iters), 'damping': damp,
               'note': 'candidate parameters; not a recovery of the reference grade'}


# ---------------------------------------------------------------------------
# optional numerical refinement on top of the analytic estimate
# ---------------------------------------------------------------------------

def _style_loss(tgt, ref, p, bins=10):
    cur = apply_params(tgt, p)
    cl, rl = luminance(cur).ravel(), luminance(ref).ravel()
    grid = np.linspace(0, 1, 129)
    lc = np.cumsum(np.histogram(cl, grid)[0]) / len(cl)
    lr = np.cumsum(np.histogram(rl, grid)[0]) / len(rl)
    lum_term = float(np.abs(lc - lr).mean())
    a = _bin_resid(cur, bins)
    b = _bin_resid(ref, bins)
    ok = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    if ok.sum():
        chroma_term = float(np.linalg.norm(a[ok] - b[ok], axis=1).mean())
    else:
        chroma_term = 0.0
    return lum_term + 0.5 * chroma_term


def refine(target, reference, p0, budget=240, seed=20260927, bins=10, side=160):
    """Coordinate-wise refinement seeded by the analytic estimate.

    Searches only in a neighbourhood of p0, so this cannot rescue a bad analytic
    estimate and is not evidence that the parametric mode is better.
    """
    tgt = _downsample(np.clip(np.asarray(target, dtype=np.float32), 0, 1), side)
    ref = _downsample(np.clip(np.asarray(reference, dtype=np.float32), 0, 1), side)
    p = clip_params(p0)
    best = _style_loss(tgt, ref, p, bins)
    scales = (_HIGH - _LOW) * np.array([.12, .12, .12, .12, .12, .3, .3, .3, .3, .3, .3])
    rng = np.random.default_rng(seed)
    for _ in range(int(budget)):
        k = int(rng.integers(0, N_PARAM))
        step = float(rng.normal() * scales[k])
        cand = clip_params(p + np.eye(N_PARAM)[k] * step)
        val = _style_loss(tgt, ref, cand, bins)
        if val < best:
            best, p = val, cand
    return p, {'loss': float(best), 'loss_before': float(_style_loss(tgt, ref, clip_params(p0), bins))}


if __name__ == '__main__':
    # Smoke test only: does the neutral parameter vector behave as a no-op?
    rng = np.random.default_rng(0)
    x = rng.random((32, 48, 3)).astype(np.float32)
    d = np.abs(apply_params(x, neutral()) - x).max()
    print('neutral max deviation:', float(d))
    print('param count stored:', N_PARAM, ' effective: 9')
    print('OK' if d < 2e-6 else 'FAIL')
