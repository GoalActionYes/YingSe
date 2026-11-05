"""Fast capacity test for the parametric mode.

Question it answers: does ANY parameter vector inside mode B's box beat what the
existing preset-blend mode already produces, on the same reference and the same
objective? If yes, mode B's problem is its estimator. If no, the box itself is the
limit and the parameterisation (or its bounds) needs to change.

Objective: luminance CDF L1 + 0.5 * chroma-bin L1, computed at reduced size.
This is a distributional objective, not a perceptual one.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import compare_modes as cm
import param_match as pm

HERE = Path(__file__).resolve().parent


def obj(out, ref):
    return float(cm.lum_cdf_l1(out, ref) + 0.5 * cm.chroma_bin_l1(out, ref))


def oracle(tgt, ref, budget=1200, seed=99, passes=40):
    lo, hi = pm._LOW, pm._HIGH
    rng = np.random.default_rng(seed)
    best_p = pm.neutral()
    best_v = obj(pm.apply_params(tgt, best_p), ref)
    total = 1

    for _ in range(budget):
        cand = lo + rng.random(pm.N_PARAM) * (hi - lo)
        v = obj(pm.apply_params(tgt, cand), ref)
        total += 1
        if v < best_v:
            best_v, best_p = v, cand

    scales = (hi - lo) * 0.2
    for shrink in (1.0, 0.35, 0.12, 0.04):
        for _ in range(passes):
            moved = False
            for k in range(pm.N_PARAM):
                for s in (shrink, -shrink):
                    cand = pm.clip_params(best_p + np.eye(pm.N_PARAM)[k] * scales[k] * s)
                    v = obj(pm.apply_params(tgt, cand), ref)
                    total += 1
                    if v < best_v - 1e-8:
                        best_v, best_p, moved = v, cand, True
            if not moved:
                break
    return best_p, best_v, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='data/6a504c47a7de4a4e83cdd22dc4c71202.jpg')
    ap.add_argument('--seeds', default='11')
    ap.add_argument('--side', type=int, default=128)
    ap.add_argument('--budget', type=int, default=1200)
    ap.add_argument('--out', default=str(HERE / 'oracle_b.json'))
    args = ap.parse_args()

    full = cm.load(HERE / args.target)
    st = max(1, full.shape[0] // args.side)
    tgt = full[::st, ::st]
    print(f'size used: {tgt.shape}  (step {st})', flush=True)

    rows = []
    for sd in [int(s) for s in args.seeds.split(',')]:
        ref_full = cm.gen_reference('c1c', full, seed=sd)
        ref = ref_full[::st, ::st]

        t0 = time.perf_counter()
        best_p, best_v, n_eval = oracle(tgt, ref, args.budget)
        dt = time.perf_counter() - t0

        a_res = cm.run_mode_a(full, ref_full)
        b_res = cm.run_mode_b(full, ref_full, False, 'v2')
        v_a = obj(a_res['out'][::st, ::st], ref)
        v_b = obj(b_res['out'][::st, ::st], ref)
        v_id = obj(tgt, ref)

        verdict = 'B 天花板低于 A 实际输出 -> 参数盒本身受限' if best_v >= v_a \
            else 'B 天花板高于 A -> 问题在估计器，不在参数化'
        rows.append({'seed': sd, 'identity': round(v_id, 5), 'A_shipped': round(v_a, 5),
                     'B_shipped': round(v_b, 5), 'B_oracle': round(best_v, 5),
                     'oracle_minus_A': round(best_v - v_a, 5),
                     'estimator_gap': round(v_b - best_v, 5),
                     'evals': n_eval, 'sec': round(dt, 1),
                     'oracle_params': {k: round(v, 3) for k, v in pm.params_dict(best_p).items()},
                     'verdict': verdict})
        print(f'seed {sd}  identity={v_id:.4f}  A_shipped={v_a:.4f}  '
              f'B_v2_shipped={v_b:.4f}  B_ORACLE={best_v:.4f}  ({dt:.0f}s, {n_eval} evals)', flush=True)
        print(f'          {verdict}   (oracle - A = {best_v - v_a:+.4f};  '
              f'估计器差距 = {v_b - best_v:+.4f})', flush=True)
        print(f'          oracle 参数: {rows[-1]["oracle_params"]}', flush=True)

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
