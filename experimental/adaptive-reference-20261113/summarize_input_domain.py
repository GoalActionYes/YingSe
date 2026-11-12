"""Curated evidence for input adaptation and first holdout, no photo/model bundle."""
import hashlib
import json
from pathlib import Path
from salut_model import ROOT


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113'
    paths = {
        'domain_forward': 'pst50/development-domain-probe/results.json',
        'inverse_pst50': 'pst50/development-salut-inverse-v2/results.json',
        'inverse_coco_development': 'independent-coco/development/salut-inverse/results.json',
        'wct2_development': 'pst50/development-wct2/results.json',
        'first_holdout': 'independent-coco/evaluation/salut-inverse-first-holdout/results.json',
        'repaired_validation': 'independent-coco/evaluation/salut-inverse-roundoff-repair/results.json',
    }
    records = {key: json.loads((base / path).read_text(encoding='utf-8')) for key, path in paths.items()}
    first, repaired = records['first_holdout'], records['repaired_validation']
    unchanged = [(old, new) for old, new in zip(first['cases'], repaired['cases']) if 'error' not in old]
    if len(unchanged) != 115 or any(old != new for old, new in unchanged):
        raise ValueError('Successful first-holdout rows were changed')
    first_dir = base / 'independent-coco/evaluation/salut-inverse-first-holdout'
    repair_dir = base / 'independent-coco/evaluation/salut-inverse-roundoff-repair'
    first_lock = json.loads((first_dir / 'run-lock.json').read_text(encoding='utf-8'))
    repair_lock = json.loads((repair_dir / 'repair-lock.json').read_text(encoding='utf-8'))
    evidence = {
        'scope': 'Research validation failed product admission; post-holdout numerical repair explicitly separated',
        'raw_report_sha256': {key: hashlib.sha256((base / path).read_bytes()).hexdigest() for key, path in paths.items()},
        'domain_forward_cases': records['domain_forward']['cases'],
        'inverse_pst50': records['inverse_pst50'],
        'inverse_coco_development': records['inverse_coco_development'],
        'wct2_development': records['wct2_development'],
        'first_holdout': {key: first[key] for key in ('scope', 'run_lock_sha256', 'fixed_inverse_fit', 'gate', 'summary', 'by_profile', 'by_group')},
        'first_failed_cases': [r for r in first['cases'] if 'error' in r],
        'first_frozen_source_sha256': first_lock['research_sha256'],
        'repair_provenance': repair_lock,
        'repaired_validation': {key: repaired[key] for key in ('scope', 'repair_lock_sha256', 'fixed_inverse_fit', 'gate', 'summary', 'by_profile', 'by_group')},
        'repaired_case_scores': [{key: r[key] for key in ('target', 'reference', 'profile', 'group', 'baseline', 'canonical', 'salut_inverse')}
                                 for r in repaired['cases']],
        'unchanged_successful_first_rows_verified': 115,
        'experimental_unit_checks_passed': 57,
        'product_integration': False,
        'heldout_status': {'original_coco_24_pairs_consumed': True, 'pst50_pairs_6_to_50_consumed': False},
    }
    names = ('salut_domain.py', 'salut_holdout.py', 'repair_salut_holdout_roundoff.py', 'wct2_model.py',
             'salut-domain-protocol.md', 'salut-holdout-protocol.md', 'wct2-protocol.md')
    evidence['current_research_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    destination = ROOT / 'docs/research/2026-11-13-input-domain-and-holdout-evidence.json'
    destination.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(destination.name)


if __name__ == '__main__':
    main()
