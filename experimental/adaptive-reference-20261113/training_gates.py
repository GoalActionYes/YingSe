"""Development gates only; never substitutes for independent/visual testing."""
import math
import numpy as np


def _scores(report,count):
    rows=report['cases']
    if len(rows)!=count:raise ValueError('Incomplete development scores')
    values={row['index']:float(row['mae']) for row in rows}
    if len(values)!=count or any(not math.isfinite(v) or v<0 for v in values.values()):raise ValueError('Invalid development scores')
    return values


def development_gate(baseline,selected,expanded=False,control=None):
    count=64 if expanded else 32;before=_scores(baseline,count);after=_scores(selected,count)
    if set(before)!=set(after):raise ValueError('Development sample identities differ')
    mean_before=float(np.mean(list(before.values())));mean_after=float(np.mean(list(after.values())))
    improved=sum(before[k]-after[k]>.1 for k in before)
    severe=sum(after[k]-before[k]>5 and after[k]>before[k]*1.25 for k in before)
    improvement=1-mean_after/mean_before if mean_before>0 else 0.
    gate=dict(relative_improvement=improvement,improved=improved,severe_regressions=severe)
    gate['proceed_to_independent_evaluation']=improvement>=.05 and improved>=(40 if expanded else 20) and severe<=(2 if expanded else 1)
    if expanded:
        added=[r['index'] for r in baseline['cases'][32:]]
        subgroup=dict(baseline_mean=float(np.mean([before[k] for k in added])),selected_mean=float(np.mean([after[k] for k in added])),improved=sum(before[k]-after[k]>.1 for k in added))
        gate['added_validation']=subgroup
        gate['proceed_to_independent_evaluation'] &= subgroup['selected_mean']<=subgroup['baseline_mean'] and subgroup['improved']>=17
    if control is not None:
        comparison=_scores(control,count)
        if set(comparison)!=set(before):raise ValueError('Control sample identities differ')
        control_mean=float(np.mean(list(comparison.values())))
        gate.update(relative_improvement_vs_control=1-mean_after/control_mean if control_mean>0 else 0.,
            improved_vs_control=sum(comparison[k]-after[k]>.1 for k in before),
            severe_vs_control=sum(after[k]-comparison[k]>5 and after[k]>comparison[k]*1.25 for k in before))
        gate['proceed_to_independent_evaluation'] &= gate['relative_improvement_vs_control']>=.03 and gate['improved_vs_control']>=(36 if expanded else 18) and gate['severe_vs_control']<=(2 if expanded else 1)
    return gate


def reserved_gate(cases,duplicates=()):
    valid=[r for r in cases if 'scores' in r]
    complete=len(cases)==32 and len(valid)==32 and len({r['index'] for r in valid})==32
    gate=dict(scored=len(valid),total=32,complete=complete,no_exact_overlap=not duplicates,passed=False)
    if not valid:return gate
    actual=np.array([r['scores']['candidate'] for r in valid],dtype=float)
    for name,label in [('before','primary'),('current','legacy_global')]:
        baseline=np.array([r['scores'][name] for r in valid],dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(baseline).all() or (actual<0).any() or (baseline<0).any():raise ValueError('Invalid reserved scores')
        mean=float(baseline.mean());gate[label]=dict(baseline_mean=mean,candidate_mean=float(actual.mean()),
            relative_improvement=1-float(actual.mean())/mean if mean>0 else 0.,improved=int((actual<baseline-.1).sum()),
            regressed=int((actual>baseline+.1).sum()),severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum()))
    primary,legacy=gate['primary'],gate['legacy_global']
    gate['passed']=complete and not duplicates and primary['relative_improvement']>=.05 and primary['improved']>=20 and primary['severe']<=1 and legacy['relative_improvement']>=.10 and legacy['improved']>=20 and legacy['severe']<=2
    return gate
