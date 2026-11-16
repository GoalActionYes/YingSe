"""Check historic data exposure; no new inference, scoring or split changes."""
import json
from pathlib import Path
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    load=lambda p:json.loads(p.read_text())
    manifest=load(base/'manifest.json');jobs=load(base/'evaluation/jobs.json')
    first=base/'evaluation/salut-inverse-first-holdout'
    repair=base/'evaluation/salut-inverse-roundoff-repair'
    original=load(first/'results.json');merged=load(repair/'results.json')
    assert digest(first/'run-lock.json')==original['run_lock_sha256']
    assert digest(repair/'repair-lock.json')==merged['repair_lock_sha256']
    pairs={(r['target'],r['reference']) for r in manifest['pairs'] if r['split']=='evaluation'}
    development={(r['target'],r['reference']) for r in manifest['pairs'] if r['split']=='development'}
    assert len(pairs)==24 and len(development)==6 and not pairs&development
    keys=lambda rows:{(r['target'],r['reference'],r['profile']) for r in rows}
    expected=keys(jobs['jobs']);actual=keys(merged['cases']);profiles={r['profile'] for r in jobs['jobs']}
    assert len(profiles)==5 and expected==actual and len(actual)==len(merged['cases'])==120
    assert actual=={(a,b,p) for a,b in pairs for p in profiles}
    assert merged['gate']['complete'] and merged['gate']['valid']==120 and merged['gate']['passed'] is False
    paths=[base/'manifest.json',base/'evaluation/jobs.json',first/'results.json',first/'run-lock.json',repair/'results.json',repair/'repair-lock.json']
    report=dict(scope='Historic exposure audit only; no new inference or score recomputation',
        sources_sha256={str(p.relative_to(base)):digest(p) for p in paths},
        evaluation_pairs_consumed=24,controls_per_pair=5,scored_rows=120,
        original_valid=original['gate']['valid'],repaired_valid=merged['gate']['valid'],
        no_missing_or_duplicate_jobs=True,promotion_passed=False,eligible_as_fresh_independent_test=False,
        pairs=[dict(target=a,reference=b) for a,b in sorted(pairs)],source_sha256=digest(Path(__file__)))
    (ROOT/'docs/research/2026-11-17-coco-consumption-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['pairs','sources_sha256']}))


if __name__=='__main__':main()
