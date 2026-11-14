"""Complete grouped passes for a fixed equal-update training block."""
from paired_training import grouped_batches


def block_batches(rows,rng,passes):
    if type(passes) is not int or passes<1:raise ValueError('Expected positive whole passes')
    return [batch for _ in range(passes) for batch in grouped_batches(rows,rng)]


def combined_gate(matched,original,expanded):
    a=matched.get('proceed_to_independent_evaluation');b=original.get('proceed_to_independent_evaluation')
    if type(a) is not bool or type(b) is not bool or type(expanded) is not bool:
        raise ValueError('Expected explicit complete-gate booleans')
    return dict(matched,matched_control_gate_passed=a,original_C10_gate_passed=b,
                expanded_data_candidate=expanded,proceed_to_independent_evaluation=expanded and a and b)
