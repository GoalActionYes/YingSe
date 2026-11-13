"""Unchanged primary qualification plus predeclared strong-control nonregression."""
import numpy as np
from pairwise_gate import response_development_gate


def invariance_gate(baseline,candidate,c00,c10):
    primary=response_development_gate(baseline,candidate,c00)
    response_development_gate(baseline,c10,c00)  # Validate identities/finite per-case and paired data.
    mean=lambda report:float(np.mean([r['mae'] for r in report['cases']]))
    delta=lambda report:float(np.mean([r['delta_mae'] for r in report['paired']]))
    c,a,cd,ad=mean(candidate),mean(c10),delta(candidate),delta(c10)
    secondary=dict(candidate_mean=c,appearance_mean=a,candidate_delta_mean=cd,appearance_delta_mean=ad,rgb_no_worse=c<=a,delta_no_worse=cd<=ad)
    return dict(primary,primary_gate_passed=primary['proceed_to_independent_evaluation'],c10_nonregression=secondary,
        proceed_to_independent_evaluation=bool(primary['proceed_to_independent_evaluation'] and secondary['rgb_no_worse'] and secondary['delta_no_worse']))
