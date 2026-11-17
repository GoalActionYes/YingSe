"""Freeze a cross-role, source-family-disjoint split for reference swap experiments."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

from audit_tst_reference_equivalence import COHORT, HASHES, audit, family, load


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paths(case):
    return [case['content'], *case['refs'], case['same_truth'], case['different_truth']]


def split(out, min_conflict=False):
    metadata, cases, ready = audit(limit=None)
    cohort = load(COHORT, HASHES[1])
    pilot_path = COHORT.parent.parent / 'reference-equivalence-pilot-v1/private-selection.json'
    pilot_evidence = COHORT.parents[4] / 'docs/research/2026-11-18-tst-reference-equivalence-evidence.json'
    pilot_report = json.loads(pilot_evidence.read_text(encoding='utf-8'))
    pilot = load(pilot_path, pilot_report['selection_sha256'])
    assert cases[:8] == pilot and len(pilot) == 8

    def source(path):
        return family(path, 'content' if path.startswith('content_images/') else 'reference')

    parents = {}

    def root(label):
        while label in parents:
            label = parents[label]
        return label

    def connect(left, right):
        a, b = sorted((root(left), root(right)))
        if a != b:
            parents[b] = a

    for left, right in cohort['alias_pairs']:
        connect(left, right)
    assets = sorted({p for case in cases for p in paths(case)})
    for path in assets:
        entry = ready[path]
        assert digest(COHORT.parent / entry['prepared_path']) == entry['sha256']
    edges = Counter()
    for left, right in combinations(assets, 2):
        a, b = source(left), source(right)
        if a == b:
            continue
        x, y = ready[left], ready[right]
        if any(x[key] == y[key] for key in ('original_sha256', 'pixel_sha256')):
            edges['exact'] += 1
        elif (x['perceptual_hash'] ^ y['perceptual_hash']).bit_count() <= 6:
            edges['near'] += 1
        else:
            continue
        connect(a, b)

    def groups(case):
        return {root(source(p)) for p in paths(case)}

    seen_pilot = set().union(*(groups(case) for case in pilot))
    remain = []
    discarded = Counter()
    for case in cases:
        if len({root(source(p)) for p in paths(case)[:4]}) != 4:
            discarded['within_case_source_overlap'] += 1
        elif groups(case) & seen_pilot:
            discarded['pilot_overlap'] += 1
        else:
            remain.append(case)
    remain.sort(key=lambda case: hashlib.sha256(('reference-equivalence-split-v1:' + case['content']).encode()).hexdigest())
    development = []
    seen_development = set()
    group_sets = [groups(case) for case in remain]
    available = list(range(len(remain)))
    while available and len(development) < 24:
        if min_conflict:
            index = min(available, key=lambda i: sum(not group_sets[i].isdisjoint(group_sets[j]) for j in available))
        else:
            index = available[0]
        case, components = remain[index], group_sets[index]
        development.append(case)
        seen_development.update(components)
        available = [other for other in available if components.isdisjoint(group_sets[other])]
    training = [case for case in remain if not groups(case) & seen_development]
    seen_train = set().union(*(groups(case) for case in training))
    assert not (seen_train & seen_development or seen_train & seen_pilot or seen_development & seen_pilot)
    assert len(development) + len(training) <= len(remain)
    private = dict(development=development, training=training)
    out.mkdir(exist_ok=False)
    assignment = out / 'private-assignment.json'
    assignment.write_text(json.dumps(private, indent=2), encoding='utf-8')
    result = dict(metadata=metadata, candidate_cases=len(cases), verified_prepared_assets=len(assets),
                  content_edges=dict(edges), dropped_before_split=dict(discarded),
                  development_conflict_dropped=len(remain) - len(development) - len(training),
                  development_cases=len(development), training_cases=len(training),
                  development_components=len(seen_development), training_components=len(seen_train),
                  assignment_sha256=digest(assignment), pilot_sha256=digest(pilot_path),
                  protocol_sha256=digest(Path(__file__).with_name('tst-reference-equivalence-split-protocol.md')),
                  script_sha256=digest(Path(__file__)), allocator='least_remaining_conflict' if min_conflict else 'fixed_order',
                  readiness=len(development) == 24 and len(training) >= 64)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--min-conflict', action='store_true')
    args = parser.parse_args()
    split(args.out, args.min_conflict)
