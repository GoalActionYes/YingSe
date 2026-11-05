"""Side-by-side measurement of the two auto-match modes. Not a verdict generator.

Mode A  preset blend   : grading.recommend -> 9 weights -> grading.transform   (existing)
Mode B  parametric     : param_match.estimate_params -> 11 bounded parameters  (candidate)

Two evaluation conditions, because they answer different questions:

  C1 same-content   reference = a known preset grade applied to the SAME target.
                    Content is identical, so matching error is attributable to the
                    method. This is the condition that can say something about
                    coverage / expressiveness. It is still not a visual-quality
                    verdict on real footage.

  C2 cross-image    target and reference are different photographs. This is the
                    actual use case, but every distance metric here is confounded
                    by scene content: two different scenes legitimately have
                    different luminance and chroma distributions. Numbers from C2
                    are reported as side-effect and plausibility diagnostics only.

Metrics
  lum_cdf_l1        L1 between output and reference luminance CDFs (lower = closer).
                    Only interpretable in C1.
  chroma_bin_l1     mean chroma-residual distance over matched luminance bins.
                    Only interpretable in C1.
  skin_deltaE       mean CIE-Lab distance on a skin mask between output and the
                    ORIGINAL target (lower = less disturbance). Approximate mask.
  detail_dark/_bright  local-contrast retention ratio in the darkest / brightest
                    decile (1.0 = unchanged).
  band_max_step     largest 8-bit code jump along a grey ramp (lower = smoother).
  band_codes        number of distinct 8-bit codes produced over a 1024-step ramp.
  monotone_frac     fraction of LUT samples that are non-decreasing along R, G, B.
  temporal_ratio    std of per-frame mean luminance after / before, for a synthetic
                    exposure-drift sequence with ONE fixed LUT. >1 means the grade
                    amplifies variation that already exists in the footage.
  ms                wall time for estimation + one full-size application.

Usage
  python compare_modes.py                       # demo target, all presets, C1
  python compare_modes.py --cond c2 --limit 4   # cross-image pairs from data/
  python compare_modes.py --target a.jpg --reference b.jpg --cond c2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import grading
import param_match as pm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'data'
DEMO = HERE / 'demo'

LUMA_W = np.array([.2126, .7152, .0722], dtype=np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def lum(x):
    return (np.asarray(x, dtype=np.float32) * LUMA_W).sum(-1, keepdims=True)


def load(path, side=640):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((side, side))
        return np.asarray(im).astype(np.float32) / 255.


def srgb_to_lab(rgb):
    x = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]],
                 dtype=np.float32)
    xyz = (lin @ m.T) / np.array([.95047, 1.0, 1.08883], dtype=np.float32)
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lum_cdf_l1(a, b, bins=128):
    grid = np.linspace(0, 1, bins + 1)
    ya, yb = lum(a).ravel(), lum(b).ravel()
    ha = np.cumsum(np.histogram(ya, grid)[0]) / len(ya)
    hb = np.cumsum(np.histogram(yb, grid)[0]) / len(yb)
    return float(np.abs(ha - hb).mean())


def bin_resid(rgb, n=10, min_frac=0.01):
    """Mean chroma residual per fixed luminance bin; thin bins dropped."""
    x = np.asarray(rgb, dtype=np.float32)
    y = lum(x).ravel()
    idx = np.clip((y * n).astype(np.int32), 0, n - 1)
    res = (x - lum(x)).reshape(-1, 3)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for b in range(n):
        m = idx == b
        if m.sum() >= min_frac * len(idx):
            out[b] = res[m].mean(0)
    return out


def chroma_bin_l1(a, b, n=10):
    x, y = bin_resid(a, n), bin_resid(b, n)
    ok = ~np.isnan(x[:, 0]) & ~np.isnan(y[:, 0])
    if not ok.sum():
        return float('nan')
    return float(np.linalg.norm(x[ok] - y[ok], axis=1).mean())


def skin_mask(rgb):
    """Approximate skin mask from the ORIGINAL frame. Heuristic, not ground truth."""
    lab = srgb_to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    ang = np.arctan2(b, a)
    chroma = np.sqrt(a * a + b * b)
    return (L > 20) & (L < 95) & (ang > 0.35) & (ang < 1.05) & (chroma > 5)


def skin_delta(original, output):
    m = skin_mask(original)
    if m.sum() < 200:
        return float('nan'), int(m.sum())
    d = srgb_to_lab(output) - srgb_to_lab(original)
    return float(np.sqrt((d ** 2).sum(-1))[m].mean()), int(m.sum())


def _boxsum(x, r):
    n = 2 * r + 1
    p = np.pad(x, ((r, r), (r, r)), mode='edge')
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def detail_retention(original, output):
    """Local-contrast retention in the darkest and brightest deciles."""
    y0, y1 = lum(original)[..., 0], lum(output)[..., 0]
    s0 = np.sqrt(np.maximum(_boxsum(y0 * y0, 2) - _boxsum(y0, 2) ** 2, 0))
    s1 = np.sqrt(np.maximum(_boxsum(y1 * y1, 2) - _boxsum(y1, 2) ** 2, 0))
    q = np.quantile(y0, [0.10, 0.90])
    out = {}
    for name, m in (('dark', y0 <= q[0]), ('bright', y0 >= q[1])):
        if m.sum() < 200:
            out[name] = float('nan')
            continue
        base = float(s0[m].mean())
        out[name] = float(s1[m].mean() / base) if base > 1e-8 else float('nan')
    return out


def ramp_metrics(lut, n=1024):
    """Banding diagnostics on a grey ramp, measured through the real LUT path."""
    ramp = np.tile(np.linspace(0, 1, n, dtype=np.float32), (8, 1))[..., None]
    ramp = np.repeat(ramp, 3, -1)
    out = grading.apply_lut(ramp, lut) if lut.ndim == 4 else pm.apply_lut(ramp, lut)
    codes = np.round(np.clip(out[0, :, 0], 0, 1) * 255)
    d = np.abs(np.diff(codes))
    return {'band_max_step': int(d.max()) if len(d) else 0,
            'band_codes': int(len(np.unique(codes)))}


def tone_mono_frac(lut):
    """Monotonicity along the grey diagonal. This is the response that actually
    governs tone inversion on a neutral ramp."""
    n = len(lut)
    idx = np.arange(n)
    diag = lut[idx, idx, idx]                 # [n, 3] along grey axis
    d = np.diff(diag, axis=0)
    return float(1 - int((d < -1e-6).sum()) / max(d.size, 1))


def chan_mono_frac(lut):
    """Per-channel monotonicity over the whole cube.

    Reported with a caveat: saturation and split tone are cross-channel by
    construction (raising R lowers the G output when sat > 1), so this number is
    below 1 for any transform that includes them. The existing presets behave the
    same way. It is a diagnostic, not a pass/fail criterion.
    """
    tot = bad = 0
    for axis in (2, 1, 0):          # red, green, blue
        d = np.diff(lut, axis=axis)
        tot += d.size
        bad += int((d < -1e-6).sum())
    return float(1 - bad / max(tot, 1))


def _pchip(xn, xs, ys):
    """Monotone cubic Hermite (Fritsch-Carlson). Used only to build reference
    grades; it removes derivative discontinuities in the generator, which is not
    the same as removing banding from a finished 8-bit pipeline."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(len(xs) - 1):
        if delta[i] == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9:
                t = 3 / np.sqrt(s)
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]
    xn = np.clip(np.asarray(xn, np.float64), xs[0], xs[-1])
    k = np.clip(np.searchsorted(xs, xn) - 1, 0, len(xs) - 2)
    t = (xn - xs[k]) / h[k]
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * ys[k] + h10 * h[k] * m[k] + h01 * ys[k + 1] + h11 * h[k] * m[k + 1]


def independent_grade(x, rng):
    """A grade family that NEITHER mode was built to represent: per-channel
    gain and gamma, a five-knot monotone cubic tone curve, split tone, saturation.

    Used to generate C1c references so neither mode gets a home advantage.
    """
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    gain = rng.uniform(0.78, 1.22, 3).astype(np.float32)
    gamma = rng.uniform(0.78, 1.32, 3).astype(np.float32)
    knots = np.sort(rng.uniform(0.02, 0.98, 5))
    xs = np.concatenate([[0.], knots, [1.]])
    ys = np.sort(rng.uniform(0.02, 0.98, 5))
    ys = np.concatenate([[0.], ys, [1.]])
    sh = rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    hi = rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    sat = float(rng.uniform(0.65, 1.45))

    out = np.clip((x * gain) ** gamma, 0, 1)
    out = np.clip(_pchip(out, xs, ys), 0, 1).astype(np.float32)
    y = lum(out)
    out = np.clip(out + (1 - y) * sh + y * hi, 0, 1)
    y2 = lum(out)
    return np.clip(y2 + (out - y2) * sat, 0, 1).astype(np.float32)


def temporal_ratio(target, lut, frames=6, drift=0.02):
    """Amplification of existing exposure variation under ONE fixed LUT."""
    seq = [np.clip(target * (1 + drift * k), 0, 1) for k in range(frames)]
    means_in = np.array([float(lum(f).mean()) for f in seq])
    outs = [grading.apply_lut(f, lut) for f in seq]
    means_out = np.array([float(lum(o).mean()) for o in outs])
    si = float(means_in.std())
    return float(means_out.std() / si) if si > 1e-9 else float('nan')


# ---------------------------------------------------------------------------
# the two modes
# ---------------------------------------------------------------------------

def run_mode_a(target, reference):
    t0 = time.perf_counter()
    weights, info = grading.recommend(target, reference)
    w, strength = grading.split_strength(weights)
    lut = grading.make_lut(w, strength)
    out = grading.apply_lut(target, lut)
    ms = (time.perf_counter() - t0) * 1000
    return {'out': out, 'lut': lut, 'ms': ms, 'params': {'weights': [round(float(v), 4) for v in weights],
            'strength': round(float(strength), 4)}, 'info': info}


def run_mode_b(target, reference, refine=False, estimator='v2'):
    t0 = time.perf_counter()
    est = pm.estimate_params_v2 if estimator == 'v2' else pm.estimate_params
    p, info = est(target, reference)
    if refine:
        p, rinfo = pm.refine(target, reference, p)
        info.update(rinfo)
    lut = pm.params_to_lut(p)
    out = pm.apply_lut(target, lut)
    ms = (time.perf_counter() - t0) * 1000
    return {'out': out, 'lut': lut, 'ms': ms,
            'params': {k: round(v, 4) for k, v in pm.params_dict(p).items()}, 'info': info}


def score(target, reference, result, cond):
    out, lut = result['out'], result['lut']
    s = {'ms': round(result['ms'], 1)}
    # distance metrics are only meaningful when content matches
    if cond == 'c1':
        s['lum_cdf_l1'] = round(lum_cdf_l1(out, reference), 5)
        s['chroma_bin_l1'] = round(chroma_bin_l1(out, reference), 5)
    else:
        s['lum_cdf_l1'] = None
        s['chroma_bin_l1'] = None
    d, n = skin_delta(target, out)
    s['skin_deltaE'] = None if d != d else round(d, 3)
    s['skin_px'] = n
    det = detail_retention(target, out)
    s['detail_dark'] = None if det['dark'] != det['dark'] else round(det['dark'], 4)
    s['detail_bright'] = None if det['bright'] != det['bright'] else round(det['bright'], 4)
    s.update(ramp_metrics(lut))
    s['tone_mono'] = round(tone_mono_frac(lut), 4)
    s['chan_mono'] = round(chan_mono_frac(lut), 4)
    s['temporal_ratio'] = round(temporal_ratio(target, lut), 4)
    return s


def identity_row(target, reference, cond):
    lut = grading.make_lut([1] + [0] * 8, 1.0)
    res = {'out': target, 'lut': lut, 'ms': 0.0, 'params': {}, 'info': {}}
    return score(target, reference, res, cond)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def gen_reference(cond, tgt, seed=11, style=1):
    """Rebuild a reference for a given condition, used for the visual strip."""
    rng = np.random.default_rng(seed)
    if cond == 'c1':
        onehot = [0.] * 9
        onehot[style] = 1.
        return grading.apply_lut(tgt, grading.make_lut(onehot, 1.0))
    if cond == 'c1b':
        p = pm.clip_params(pm.neutral() + rng.normal(scale=0.6, size=pm.N_PARAM)
                           * (pm._HIGH - pm._LOW) * 0.35)
        return pm.apply_params(tgt, p)
    if cond == 'c1c':
        return independent_grade(tgt, rng)
    raise ValueError(cond)


def cond_c1(targets, style_indices, refine, estimator='v2'):
    rows = []
    for tp in targets:
        tgt = load(tp)
        for si in style_indices:
            onehot = [0.0] * 9
            onehot[si] = 1.0
            reference = grading.apply_lut(tgt, grading.make_lut(onehot, 1.0))
            label = f'{Path(tp).stem[:18]} / {grading.LABELS[si]}'
            rows.append(('identity', label, identity_row(tgt, reference, 'c1'), {}))
            a = run_mode_a(tgt, reference)
            rows.append(('A preset-blend', label, score(tgt, reference, a, 'c1'), a['params']))
            b = run_mode_b(tgt, reference, refine, estimator)
            rows.append((f'B {estimator} parametric', label, score(tgt, reference, b, 'c1'), b['params']))
            print(f'  c1 {label}')
    return rows


def cond_c1b(targets, seeds, refine, estimator='v2'):
    """Reference = a grade drawn from mode B's own forward model.

    Symmetric counterpart to C1. C1 lets mode A match references built from its
    own basis, which gives it a home advantage; this one does the same for B.
    Neither alone decides anything - they have to be read together.
    """
    rows = []
    for tp in targets:
        tgt = load(tp)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            p = pm.clip_params(pm.neutral() + rng.normal(scale=0.6, size=pm.N_PARAM)
                               * (pm._HIGH - pm._LOW) * 0.35)
            reference = pm.apply_params(tgt, p)
            label = f'{Path(tp).stem[:16]} / B-gen s{sd}'
            rows.append(('identity', label, identity_row(tgt, reference, 'c1'), {}))
            a = run_mode_a(tgt, reference)
            rows.append(('A preset-blend', label, score(tgt, reference, a, 'c1'), a['params']))
            b = run_mode_b(tgt, reference, refine, estimator)
            rows.append((f'B {estimator} parametric', label, score(tgt, reference, b, 'c1'), b['params']))
            print(f'  c1b {label}')
    return rows


def cond_c1c(targets, seeds, refine, estimator='v2'):
    """Reference = a grade from an independent family neither mode was built for.

    This is the condition closest to the real use case and the only one of the
    three that is not tilted toward either mode.
    """
    rows = []
    for tp in targets:
        tgt = load(tp)
        for sd in seeds:
            rng = np.random.default_rng(sd)
            reference = independent_grade(tgt, rng)
            label = f'{Path(tp).stem[:16]} / indep s{sd}'
            rows.append(('identity', label, identity_row(tgt, reference, 'c1'), {}))
            a = run_mode_a(tgt, reference)
            rows.append(('A preset-blend', label, score(tgt, reference, a, 'c1'), a['params']))
            b = run_mode_b(tgt, reference, refine, estimator)
            rows.append((f'B {estimator} parametric', label, score(tgt, reference, b, 'c1'), b['params']))
            print(f'  c1c {label}')
    return rows


def cond_c2(pairs, refine, estimator='v2'):
    rows = []
    for tp, rp in pairs:
        tgt, ref = load(tp), load(rp)
        label = f'{Path(tp).stem[:16]} -> {Path(rp).stem[:16]}'
        rows.append(('identity', label, identity_row(tgt, ref, 'c2'), {}))
        a = run_mode_a(tgt, ref)
        rows.append(('A preset-blend', label, score(tgt, ref, a, 'c2'), a['params']))
        b = run_mode_b(tgt, ref, refine, estimator)
        rows.append((f'B {estimator} parametric', label, score(tgt, ref, b, 'c2'), b['params']))
        print(f'  c2 {label}')
    return rows


def pick_images(limit):
    exts = ('*.jpg', '*.jpeg', '*.png')
    seen, out = set(), []
    for e in exts:
        for p in sorted(DATA.glob(e)):
            if p.stat().st_size < 200_000:      # skip generated screenshots
                continue
            key = p.stat().st_size
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out[:limit]


def save_strip(target, reference, outs, path, labels=None):
    """Visual side-by-side: original | reference | each mode.

    Metrics cannot substitute for looking at these, and none of the numbers in
    this file are a perceptual measure.
    """
    cols = [target, reference] + outs
    head = ['original', 'reference'] + (labels or [f'mode{i}' for i in range(len(outs))])
    h = min(c.shape[0] for c in cols)
    w = min(c.shape[1] for c in cols)
    cols = [c[:h, :w] for c in cols]
    gap = 6
    canvas = np.ones((h + 26, (w + gap) * len(cols) - gap, 3), np.float32)
    x = 0
    for c in cols:
        canvas[26:26 + h, x:x + w] = np.clip(c, 0, 1)
        x += w + gap
    img = Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='c1', choices=['c1', 'c1b', 'c1c', 'c2'])
    ap.add_argument('--target', default=None)
    ap.add_argument('--reference', default=None)
    ap.add_argument('--limit', type=int, default=3)
    ap.add_argument('--styles', default='1,2,3,4,5,6,7,8')
    ap.add_argument('--seeds', default='11,12,13')
    ap.add_argument('--refine', action='store_true')
    ap.add_argument('--estimator', default='v2', choices=['v1', 'v2'])
    ap.add_argument('--strip', default=None, help='write a visual side-by-side PNG')
    ap.add_argument('--out', default=str(HERE / 'comparison.json'))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    styles = [int(s) for s in args.styles.split(',')]
    if args.cond in ('c1', 'c1b', 'c1c'):
        targets = [Path(args.target)] if args.target else [DEMO / 'source.png']
    if args.cond == 'c1':
        print('C1 same-content: reference = a preset from mode A\'s own basis.')
        print('   NOTE: this condition favours mode A by construction.')
        rows = cond_c1(targets, styles, args.refine, args.estimator)
    elif args.cond == 'c1b':
        print('C1b same-content: reference drawn from mode B\'s own forward model.')
        print('   NOTE: this condition favours mode B by construction.')
        rows = cond_c1b(targets, seeds, args.refine, args.estimator)
    elif args.cond == 'c1c':
        print('C1c same-content: reference from an independent grade family.')
        print('   NOTE: the only condition not tilted toward either mode.')
        rows = cond_c1c(targets, seeds, args.refine, args.estimator)
    else:
        if args.target and args.reference:
            pairs = [(args.target, args.reference)]
        else:
            imgs = pick_images(max(2, args.limit * 2))
            pairs = [(imgs[i], imgs[i + 1]) for i in range(len(imgs) - 1)][:args.limit]
        print('C2 cross-image: content differs, distance metrics are confounded')
        rows = cond_c2(pairs, args.refine, args.estimator)

    cols = ['lum_cdf_l1', 'chroma_bin_l1', 'skin_deltaE', 'detail_dark', 'detail_bright',
            'band_max_step', 'band_codes', 'tone_mono', 'chan_mono', 'temporal_ratio', 'ms']
    print('\n' + 'mode'.ljust(16) + 'case'.ljust(34) + ''.join(c[:11].rjust(13) for c in cols))
    for mode, label, s, _ in rows:
        cells = []
        for c in cols:
            v = s.get(c)
            cells.append('-' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v)))
        print(mode.ljust(16) + label[:33].ljust(34) + ''.join(x.rjust(13) for x in cells))

    Path(args.out).write_text(json.dumps(
        [{'mode': m, 'case': l, 'scores': s, 'params': p} for m, l, s, p in rows],
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {args.out}')

    if args.strip:
        if args.cond == 'c2':
            tgt, ref = load(pairs[0][0]), load(pairs[0][1])
        else:
            tgt = load(targets[0])
            ref = gen_reference(args.cond, tgt, seeds[0], styles[0] if args.cond == 'c1' else 1)
        a = run_mode_a(tgt, ref)
        b = run_mode_b(tgt, ref, args.refine, args.estimator)
        path, head = save_strip(tgt, ref, [a['out'], b['out']], args.strip,
                                ['A preset-blend', f'B {args.estimator}'])
        print(f'wrote {path}  ({" | ".join(head)})')


if __name__ == '__main__':
    main()
