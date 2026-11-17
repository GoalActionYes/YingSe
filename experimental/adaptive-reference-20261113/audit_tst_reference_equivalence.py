"""Find author-indexed reference swaps with unchanged targets from prepared training assets."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / '.tools/TST100K-index/triplet.json'
COHORT = ROOT / 'outputs/adaptive-reference-20261113/tst100k/cohort-training-coverage-v2/cohort.json'
MANIFEST = COHORT.parent / 'srgb-v1/manifest.json'
HASHES = (
    '5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413',
    'ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077',
    'da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f',
)
ROLES = ('content', 'reference', 'gt')


def load(path, expected):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, path
    return json.loads(data)


def family(path, role):
    stem = PurePosixPath(path).stem
    return stem if role == 'content' else stem.rsplit('_', 1)[0]


def group_references(rows, parent, ready=None):
    answers = defaultdict(set)
    for row in rows:
        answers[row['content'], row['reference']].add(row['gt'])
    values = defaultdict(dict)
    for (content, reference), results in answers.items():
        if len(results) != 1:
            continue
        gt, = results
        if ready is not None and not {content, reference, gt} <= ready:
            continue
        target_family = parent(family(content, 'content'))
        reference_family = parent(family(reference, 'reference'))
        if target_family != reference_family:
            values[content, gt][reference_family] = min(reference, values[content, gt].get(reference_family, reference))
    return values


def count_groups(values, parent):
    alternatives = defaultdict(list)
    for (content, _), reference_families in values.items():
        alternatives[content].append(len(reference_families))
    multiple = {content for content, counts in alternatives.items() if max(counts) >= 2}
    matched = {parent(family(content, 'content')) for content in multiple if len(alternatives[content]) >= 2}
    return dict(target_result_groups=len(values),
                same_result_with_multiple_reference_families=sum(len(v) >= 2 for v in values.values()),
                targets_with_reference_substitutes=len(multiple),
                independent_target_families_with_alternative_result=len(matched))


def frozen_cases(grouped, parent):
    per_target = defaultdict(dict)
    for (content, answer), references in grouped.items():
        per_target[content][answer] = references
    chosen = []
    seen = set()
    for content, answers in sorted(per_target.items()):
        target_family = parent(family(content, 'content'))
        if target_family in seen:
            continue
        selection = next((dict(content=content, same_truth=same_truth, different_truth=different_truth,
                               refs=[a[1], b[1], c[1]])
                          for same_truth, refs in sorted(answers.items()) if len(refs) >= 2
                          for a_index, a in enumerate(sorted(refs.items()))
                          for b in sorted(refs.items())[a_index + 1:]
                          for different_truth, different_refs in sorted(answers.items()) if different_truth != same_truth
                          for c in sorted(different_refs.items()) if c[0] not in (a[0], b[0])), None)
        if selection is not None:
            chosen.append(selection)
            seen.add(target_family)
        if len(chosen) == 8:
            break
    return chosen


def audit():
    rows, cohort, manifest = (load(path, sha) for path, sha in zip((INDEX, COHORT, MANIFEST), HASHES, strict=True))
    train = cohort['groups']['train']
    assert len(rows) == 108808 and len(train) == 4096 and manifest['passed']
    assets = {r[k] for r in train for k in ROLES}
    ready = {row['original_path']: row for row in manifest['files']
             if row['original_path'] in assets and (COHORT.parent / row['prepared_path']).is_file()}
    assert len(ready) == manifest['training_unique_assets'] == len(assets)
    parents = {}

    def parent(name):
        while name in parents:
            name = parents[name]
        return name

    for first, second in cohort['alias_pairs']:
        a, b = parent(first), parent(second)
        if a != b:
            parents[max(a, b)] = min(a, b)
    training_families = {parent(family(row[k], k)) for row in train for k in ('content', 'reference')}
    protected = {parent(family(row[k], k)) for row in cohort['groups']['validation'] for k in ('content', 'reference')}
    assert not training_families & protected
    eligible = [row for row in rows if all(parent(family(row[k], k)) in training_families for k in ('content', 'reference'))]
    # Exclude ambiguity against the full index, even when an alternate result is not prepared.
    pairs = defaultdict(set)
    for row in rows:
        pairs[row['content'], row['reference']].add(row['gt'])
    safe = [row for row in eligible if len(pairs[row['content'], row['reference']]) == 1]
    all_groups, ready_groups = group_references(safe, parent), group_references(safe, parent, ready.keys())
    report = dict(index_rows=len(rows), training_rows=len(train), ready_assets=len(ready),
                  training_family_index_rows=len(eligible), excluded_ambiguous_rows=len(eligible)-len(safe),
                  all_known_training_families=count_groups(all_groups, parent), all_assets_ready=count_groups(ready_groups, parent),
                  input_sha256=dict(index=HASHES[0], cohort=HASHES[1], manifest=HASHES[2]),
                  proceed_threshold=8)
    report['proceed'] = report['all_assets_ready']['independent_target_families_with_alternative_result'] >= 8
    return report, frozen_cases(ready_groups, parent), ready


def main(out):
    report, _, _ = audit()
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
