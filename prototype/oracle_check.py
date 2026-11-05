"""Separate estimation quality from representational capacity.

If mode B loses, there are two very different explanations and they lead to
opposite decisions:

  (a) the estimator is bad but the parameterisation can express the target
      -> keep the parameterisation, fix the estimator
  (b) even the best parameter vector inside the box cannot reach the target
      -> the parameterisation / bounds are the limit

This script measures both, by searching each mode with a large budget under the
SAME objective, and reporting the reachable minimum next to what the shipped
estimators actually produce.

Objective is the same for both modes: luminance CDF L1 + 0.5 * chroma-bin L1.
It is a distributional objective, not a perceptual one - see the caveat at the
bottom of the report.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import grading
import param_match as pm
import compare_modes as cm

HERE = Path(__file__).resolve().parent


def objective(tgt_s, ref_s, out_s, bins=10):
    """tgt/ref/out are already small; out is the graded small target."""
    l = cm.lum_cdf_l1(out_s, ref_s)
    c = cm.chroma_bin_l1(out_s, ref_s)
    return float(l + 0.5 * c), float(l), float(c)


def oracle_b(tgt, ref, budget=600, seed=7, side=200):
    """Large-budget search over B's parameter box under the shared objective."""
    t = cm._downsample(tgt, side) if hasattr(cm, '_downsample') else tgt
    t = tgt[::max(1, tgt.shape[0] // side), ::max(1, tgt.shape[1] // side)]
    r = ref[::max(1, ref.shape[0] // side), ::max(1, ref.shape[1] // side)]
    rng = np.random.default_rng(seed)
    lo, hi = pm._LOW, pm._HIGH

    def val(p):
        return objective(t, r, pm.apply_params(t, p))[0]

    best_p, best_v = pm.neutral(), val(pm.neutral())
    for _ in range(budget):
        cand = lo + rng.random(pm.N_PARAM) * (hi - lo)
        v = val(cand)
        if v < best_v:
            best_v, best_p = v, cand
    scales = (hi - lo) * 0.25
    for shrink in (1.0, 0.4, 0.15, 0.05, 0.02):
        improved = True
        while improved:
            improved = False
            for k in range(pm.N_PARAM):
                for s in (shrink, -shrink):
                    cand = pm.clip_params(best_p + np.eye(pm.N_PARAM)[k] * scales[k] * s)
                    v = val(cand)
                    if v < best_v - 1e-7:
                        best_v, best_p, improved = v, cand, True
    return best_p, float(best_v)


def oracle_a(tgt, ref, budget=1500, seed=7, side=200):
    """Large-budget search over A's weight simplex under the shared objective."""
    t = tgt[::max(1, tgt.shape[0] // side), ::max(1, tgt.shape[1] // side)]
    r = ref[::max(1, ref.shape[0] // side), ::max(1, ref.shape[1] // side)]
    rng = np.random.default_rng(seed)

    def val(w, strength=1.0):
        return objective(t, r, grading.transform(t, w, strength))[0]

    cands = np.vstack([np.eye(9), rng.dirichlet(np.full(9, 0.5), budget)])
    vals = np.array([val(c) for c in cands])
    bw = cands[int(np.argmin(vals))]
    bv = float(vals.min())
    for step in (0.2, 0.08, 0.03, 0.01, 0.003):
        improved = True
        while improved:
            improved = False
            for i in range(9):
                for j in range(9):
                    if i == j:
                        continue
                    c = bw.copy()
                    ch = min(step, c[i])
                    c[i] -= ch
                    c[j] += ch
                    v = val(c)
                    if v < bv - 1e-7:
                        bv, bw, improved = v, c, True
    return bw, bv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='data/29e575f91b764352a7a7491a91c73b7f.jpg')
    ap.add_argument('--seeds', default='11,12,13')
    ap.add_argument('--cond', default='c1c', choices=['c1', 'c1b', 'c1c'])
    ap.add_argument('--budget-b', type=int, default=600)
    ap.add_argument('--budget-a', type=int, default=1500)
    ap.add_argument('--out', default=str(HERE / 'oracle.json'))
    args = ap.parse_args()

    tgt = cm.load(HERE / args.target)
    rows = []
    st = max(1, tgt.shape[0] // 200)
    for sd in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(sd)
        if args.cond == 'c1c':
            ref = cm.independent_grade(tgt, rng)
        elif args.cond == 'c1b':
            p = pm.clip_params(pm.neutral() + rng.normal(scale=0.6, size=pm.N_PARAM)
                               * (pm._HIGH - pm._LOW) * 0.35)
            ref = pm.apply_params(tgt, p)
        else:
            w = [0.] * 9
            w[1 + sd % 8] = 1.
            ref = grading.apply_lut(tgt, grading.make_lut(w, 1.0))

        t0 = time.perf_counter()
        pa, va = oracle_a(tgt, ref, args.budget_a)
        ta = time.perf_counter() - t0
        t0 = time.perf_counter()
        pb, vb = oracle_b(tgt, ref, args.budget_b)
        tb = time.perf_counter() - t0

        # everything scored at the same small size and with the same objective
        ts = tgt[::st, ::st]
        rs = ref[::st, ::st]

        def sc(out):
            v = objective(ts, rs, out[::st, ::st] if out.shape == tgt.shape else out)
            return [round(float(x), 5) for x in v]

        si = sc(tgt)
        sa = sc(cm.run_mode_a(tgt, ref)['out'])
        sb = sc(cm.run_mode_b(tgt, ref)['out'])
        oa = sc(grading.transform(ts, pa, 1.0))
        ob = sc(pm.apply_params(ts, pb))

        rows.append({
            'seed': sd,
            'identity': si, 'A shipped': sa, 'A oracle': oa,
            'B shipped': sb, 'B oracle': ob,
            'oracle_a_sec': round(ta, 1), 'oracle_b_sec': round(tb, 1),
            'oracle_b_params': {k: round(v, 4) for k, v in pm.params_dict(pb).items()},
        })
        print(f'seed {sd}: identity={si[0]:.4f}  A_ship={sa[0]:.4f}  A_orc={oa[0]:.4f}'
              f'  B_ship={sb[0]:.4f}  B_orc={ob[0]:.4f}')

    Path(args.out).write_text(json.dumps({'cond': args.cond, 'target': args.target,
                                          'rows': rows}, ensure_ascii=False, indent=2),
                              encoding='utf-8')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
