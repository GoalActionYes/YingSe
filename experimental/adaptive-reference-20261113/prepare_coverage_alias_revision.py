"""Freeze a new metadata cohort; keep the failed v1 and all evidence intact."""
import json
from pathlib import Path
from independent_coco import ROOT,digest
from prepare_training_coverage import EXPECTED,INDEX_SHA
from training_coverage import expand_training
from coverage_aliases import heldout_alias_closure
from prepare_triplet_cohort import family


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';v1=base/'cohort-training-coverage-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));index=ROOT/'.tools/TST100K-index/triplet.json'
    if digest(index)!=INDEX_SHA:raise ValueError('Author index changed')
    cohorts={}
    for n,sha in EXPECTED.items():
        p=base/n/'cohort.json'
        if digest(p)!=sha:raise ValueError('Prior cohort changed')
        cohorts[n]=load(p)
    old=cohorts['cohort-multiref-v1'];manifest=v1/'srgb-v1/manifest.json';screen=load(manifest);review_path=v1/'srgb-v1/private-duplicate-review.json';review=load(review_path)
    if screen['passed'] or review['training_started'] or not review['frozen_cohort_trial_stopped'] or review['manifest_sha256']!=digest(manifest) or not review['all_flags_reviewed']:raise ValueError('Failed v1/review identity not established')
    if [v['flag'] for v in review['decisions']]!=screen['cross_split_near_duplicates'] or any(v['decision']!='confirmed_near_duplicate' for v in review['decisions']):raise ValueError('Incomplete or uncertain crossing review')
    source_family=lambda p:Path(p).stem if p.startswith('content_images/') else family(p)
    pairs=[(source_family(v['flag']['train']),source_family(v['flag']['heldout'])) for v in review['decisions']]
    assay=v1/'heldout-fingerprint-assay-v1';assay_review=load(assay/'private-review.json');flags=[v for v in load(assay/'private-flags.json') if not v['same_family']]
    if assay_review['panel_sha256']!=digest(assay/'other-family-flags.jpg') or sorted(v['index'] for v in assay_review['decisions'])!=list(range(len(flags))) or any(v['decision']!='same_capture_or_near_duplicate' for v in assay_review['decisions']):raise ValueError('Incomplete consumed-family review')
    pairs.extend((source_family(v['first']),source_family(v['second'])) for v in flags)
    pairs=sorted({tuple(sorted(p)) for p in pairs});held={r[k] for c in cohorts.values() for part,rows in c['groups'].items() if part in ('validation','reserved') for r in rows for k in ('content_family','reference_family')};closed=heldout_alias_closure(pairs,held)
    sentinel=dict(groups=dict(validation=[dict(content_family=f,reference_family=f) for f in sorted(closed)]))
    train,stats=expand_training(load(index),old['groups']['train'],list(cohorts.values())+[sentinel],1024,old['audit_families_forced_to_train'])
    original_path=v1/'cohort.json';original=load(original_path);before={v['index'] for v in original['groups']['train']};after={v['index'] for v in train}
    roles=lambda rows:{r[k] for r in rows for k in ('content_family','reference_family')}
    if roles(train)&closed or [v for v in train if v['group_id'] in {r['group_id'] for r in old['groups']['train']}]!=old['groups']['train']:raise ValueError('Holdout closure/old pairing violated')
    paths={r[k] for r in train+old['groups']['validation'] for k in ('content','reference','gt')};existing={r['original_path'] for r in screen['files']}
    names=['prepare_coverage_alias_revision.py','coverage_aliases.py','training_coverage.py','prepare_triplet_cohort.py','prepare_training_coverage.py','coverage-alias-revision-protocol.md'];sources={n:digest(Path(__file__).with_name(n)) for n in names}
    evidences=[manifest,review_path,assay/'results.json',assay/'private-review.json',assay/'private-flags.json',assay/'other-family-flags.jpg',*sorted((v1/'srgb-v1/duplicate-panels').glob('*.jpg'))]
    report=dict(dataset=old['dataset'],revision=old['revision'],index_sha256=INDEX_SHA,previous_cohorts_sha256=EXPECTED,failed_v1_cohort_sha256=digest(original_path),evidence_sha256={str(p.relative_to(v1)):digest(p) for p in evidences},source_sha256=sources,alias_pairs=pairs,heldout_families_before=len(held),heldout_families_after=len(closed),additional_excluded_families=sorted(closed-held),preparation=stats,groups=dict(train=train,validation=old['groups']['validation'],reserved=[]),row_diff=dict(removed_indices=sorted(before-after),added_indices=sorted(after-before)),new_asset_paths=sorted(paths-existing),audit_families_forced_to_train=old['audit_families_forced_to_train'],training_started=False,quality_scores_used=False,new_independent_allocated=False,scope='Explicit v2 capture-alias hygiene revision; not a passing v1 continuation or proof of exhaustive independence')
    out=base/'cohort-training-coverage-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(cohort_sha256=digest(out/'cohort.json'),training_pairs=len(train),development_pairs=len(old['groups']['validation']),alias_edges=len(pairs),additional_excluded_families=len(closed-held),removed_rows=len(before-after),added_rows=len(after-before),new_asset_paths=len(paths-existing),old_pairs_preserved=True,training_started=False)),flush=True)


if __name__=='__main__':main()
