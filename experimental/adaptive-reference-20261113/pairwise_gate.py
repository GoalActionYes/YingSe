"""Keep reference response and tone relationships in the full development gate."""
import math
import re
import numpy as np
from training_gates import development_gate


def require_completed_control(record,config,source_sha256):
    if record.get('finished') is not True or record.get('config')!=config or record.get('source_sha256')!=source_sha256:raise ValueError('Incomplete or changed control implementation')
    epochs=record.get('epochs');count=config['epochs'];updates=config['max_updates']
    if not isinstance(epochs,list) or len(epochs)!=count or updates%count or any(r.get('epoch')!=i or r.get('updates')!=i*updates//count for i,r in enumerate(epochs,1)):raise ValueError('Control training budget incomplete')
    best=record.get('best_epoch');gate=record.get('development_gate');sha=record.get('selected_checkpoint_sha256','')
    if type(best)!=int or not 0<=best<=count or not isinstance(gate,dict) or type(gate.get('proceed_to_independent_evaluation'))!=bool or not isinstance(sha,str) or re.fullmatch('[0-9a-f]{64}',sha) is None:raise ValueError('Final control selection or gate missing')


def _paired(report):
    rows=report['paired']
    if len(rows)!=32 or len({r['group_id'] for r in rows})!=32:raise ValueError('Expected 32 unique paired targets')
    values={r['group_id']:r for r in rows}
    for row in rows:
        for x in [row['delta_mae'],*[row['tones']['absolute_delta_error'][k] for k in ['brightness','contrast','saturation','light_minus_dark_cb','light_minus_dark_cr']]]:
            if not math.isfinite(x) or x<0:raise ValueError('Invalid response score')
    return values


def response_development_gate(baseline,selected,control=None):
    gate=development_gate(baseline,selected,expanded=True,control=control);before,after=_paired(baseline),_paired(selected)
    if set(before)!=set(after):raise ValueError('Paired identities differ')
    def delta(rows):return float(np.mean([r['delta_mae'] for r in rows.values()]))
    def tone(rows,key):return float(np.mean([r['tones']['absolute_delta_error'][key] for r in rows.values()]))
    def relationship(rows):return (tone(rows,'light_minus_dark_cb')+tone(rows,'light_minus_dark_cr'))/2
    improvement=lambda a,b:1-b/a if a>0 else 0.
    gate.update(delta_improvement_vs_start=improvement(delta(before),delta(after)),relationship_start=relationship(before),relationship_selected=relationship(after),
        relationship_improvement_vs_start=improvement(relationship(before),relationship(after)),tone_error_start={k:tone(before,k) for k in ['brightness','contrast','saturation']},
        tone_error_selected={k:tone(after,k) for k in ['brightness','contrast','saturation']})
    passed=gate['delta_improvement_vs_start']>=.05 and gate['relationship_improvement_vs_start']>=.05 and all(tone(after,k)<=tone(before,k) for k in ['brightness','contrast','saturation'])
    if control is not None:
        comparison=_paired(control)
        if set(comparison)!=set(before):raise ValueError('Control pairing differs')
        gate.update(delta_improvement_vs_control=improvement(delta(comparison),delta(after)),delta_groups_improved_vs_control=sum(comparison[k]['delta_mae']-after[k]['delta_mae']>.1 for k in before),
            relationship_control=relationship(comparison),relationship_improvement_vs_control=improvement(relationship(comparison),relationship(after)))
        passed &= gate['delta_improvement_vs_control']>=.05 and gate['delta_groups_improved_vs_control']>=20 and gate['relationship_improvement_vs_control']>=.03
    gate['proceed_to_independent_evaluation'] &= passed
    return gate
