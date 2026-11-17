"""Add explicit different-style output response to the frozen reference agreement trial."""
import argparse
import json
from pathlib import Path

import torch

from independent_coco import ROOT, digest
from train_tst_reference_equivalence import (
    BASE, FROZEN_BASELINE, SPLIT, SPLIT_SHA, SpatialEngine, decision, evaluate, load, metrics, optimize, tensor_samples,
)

EVIDENCE = ROOT / 'docs/research/2026-11-18-tst-reference-equivalence-training-evidence.json'
PRIOR = BASE / 'reference-equivalence-training-v1'


def run(out):
    torch.set_num_threads(4)
    previous = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    assert not previous['decision']['proceed'] and previous['checked_png'] == 216
    assignment = load(SPLIT / 'private-assignment.json', SPLIT_SHA)
    assert len(assignment['training']) == 82 and len(assignment['development']) == 24
    assert previous['assignment_sha256'] == SPLIT_SHA
    control = load(PRIOR / 'control/results.json', previous['arms']['control']['results_sha256'])
    agreement = load(PRIOR / 'candidate/results.json', previous['arms']['candidate']['results_sha256'])
    assert control['cases'] == previous['cases']['control'] and agreement['cases'] == previous['cases']['candidate']
    orders = load(PRIOR / 'order.json', previous['order_sha256'])
    assert len(orders) == 3 and all(sorted(order) == list(range(82)) for order in orders)
    original = load(FROZEN_BASELINE, previous['frozen_baseline_sha256'])
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == previous['init_checkpoint_sha256']
    out.mkdir(exist_ok=False)
    baseline = evaluate(out / 'initial', SPLIT, engine)
    assert baseline['predictions'] == original['predictions'] and baseline['cases'] == original['cases']
    samples = tensor_samples(assignment['training'])
    hist = engine.model.hist_projection
    hist.requires_grad_(True)
    epochs = optimize(engine, samples, orders, .25, response_weight=.25)
    checkpoint = out / 'hist-projection.pth'
    torch.save(hist.state_dict(), checkpoint)
    candidate = evaluate(out / 'candidate', SPLIT, engine)
    verdict = decision(baseline, control, candidate)
    verdict['prior_agreement'] = metrics(agreement)
    verdict['checks']['increment_over_agreement'] = candidate['passed_cases'] >= agreement['passed_cases'] + 3
    verdict['proceed'] = all(verdict['checks'].values())
    report = dict(assignment_sha256=SPLIT_SHA, previous_evidence_sha256=digest(EVIDENCE),
                  start_checkpoint_sha256=engine.checkpoint_sha, initial_exact_outputs=72,
                  order_sha256=previous['order_sha256'], new_weight_sha256=digest(checkpoint),
                  candidate_results_sha256=digest(out / 'candidate/results.json'), epochs=epochs, decision=verdict,
                  protocol_sha256=digest(Path(__file__).with_name('tst-reference-equivalence-response-protocol.md')),
                  source_sha256={name: digest(Path(__file__).with_name(name)) for name in (
                      Path(__file__).name, 'train_tst_reference_equivalence.py', 'probe_tst_reference_equivalence.py')})
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(verdict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
