"""Predeclared gates for synthetic appearance validation, not product approval."""
from collections import Counter
import numpy as np


def assess_appearance(rows):
    keys = {(r['target'], r['reference'], r['style_index']) for r in rows}
    groups = Counter(r['group'] for r in rows)
    styles = Counter(r['style_index'] for r in rows)
    local = [r for r in rows if r['local_control']]
    same = [r for r in rows if r['same_style']]
    complete = (len(rows) == len(keys) == 150 and groups == {'single': 50, 'multiple': 50, 'none': 50}
                and styles == {i: 15 for i in range(10)} and len(local) == 50 and len(same) == 60)
    result = {'count': len(rows), 'complete': complete, 'local_count': len(local), 'same_style_count': len(same), 'passed': False}
    if not rows or not local or not same:
        return result
    before = np.array([r['input_mae'] for r in rows])
    after = np.array([r['output_mae'] for r in rows])
    rb = np.array([r['input_relation_error'] for r in local], dtype=float)
    ra = np.array([r['output_relation_error'] for r in local], dtype=float)
    if not all(np.isfinite(x).all() for x in (before, after, rb, ra)):
        return result
    same_error = float(np.mean([r['output_mae'] for r in same]))
    severe = int(((after-before > 5) & (after > before*1.25)).sum())
    gates = {'whole_mae_15_percent': after.mean() <= before.mean()*.85,
             'local_relation_15_percent': ra.mean() <= rb.mean()*.85,
             'same_style_mae_at_most_2': same_error <= 2,
             'severe_regressions_at_most_10_percent': severe/len(rows) <= .1}
    result.update(mean_input_mae=float(before.mean()), mean_output_mae=float(after.mean()),
                  mean_local_input_relation=float(rb.mean()), mean_local_output_relation=float(ra.mean()),
                  mean_same_style_mae=same_error, severe_regressions=severe, gates={k: bool(v) for k, v in gates.items()},
                  passed=bool(complete and all(gates.values())))
    return result
