"""Freeze expanded train metadata against every consumed evaluation cohort."""
import json
from pathlib import Path
from independent_coco import ROOT,digest
from training_coverage import expand_training


EXPECTED={
    'cohort-v1':'0592b2a06eedc111d35358d9b3542b081b61d49013e302b65e5f041b2527db6e',
    'cohort-v2':'988889a59ec8c6591073a9d84f3cc20955d442e20cddf003f4dfcb622198862f',
    'cohort-multiref-v1':'124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d',
    'cohort-multiref-reserved-v1':'18cb8d8ad177f2ca60cb6f0e1ab6cfa5ac08ccca4b702bebe68ea138ae7eb919',
}
INDEX_SHA='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413'


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';index=ROOT/'.tools/TST100K-index/triplet.json'
    if digest(index)!=INDEX_SHA:raise ValueError('Index changed')
    cohorts={}
    for name,sha in EXPECTED.items():
        path=base/name/'cohort.json'
        if digest(path)!=sha:raise ValueError('Previous cohort changed')
        cohorts[name]=json.loads(path.read_text(encoding='utf-8'))
    old=cohorts['cohort-multiref-v1'];rows=json.loads(index.read_text(encoding='utf-8'))
    train,stats=expand_training(rows,old['groups']['train'],list(cohorts.values()),1024,old['audit_families_forced_to_train'])
    coverage=lambda entries:dict(targets=len({r['content_family'] for r in entries}),reference_families=len({r['reference_family'] for r in entries}),
        reference_paths=len({r['reference'] for r in entries}),gt_paths=len({r['gt'] for r in entries}),
        unique_asset_paths=len({r[k] for r in entries for k in ('content','reference','gt')}))
    sources={name:digest(Path(__file__).with_name(name)) for name in ['training_coverage.py','prepare_training_coverage.py','prepare_triplet_cohort.py','training-coverage-protocol.md']}
    report=dict(dataset=old['dataset'],revision=old['revision'],index_sha256=INDEX_SHA,previous_cohorts_sha256=EXPECTED,source_sha256=sources,
        groups=dict(train=train,validation=old['groups']['validation'],reserved=[]),selected_counts=dict(train=len(train),validation=64,reserved=0),
        preparation=stats,coverage=dict(old=coverage(old['groups']['train']),expanded=coverage(train)),
        audit_families_forced_to_train=old['audit_families_forced_to_train'],all_old_train_rows_preserved=True,development_unchanged=True,
        limitation='Metadata photo-family isolation only, not exact/perceptual/scene/pretrained independence; no style suffix labels inferred',
        training_started=False,no_images_downloaded_or_read=True)
    out=base/'cohort-training-coverage-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in sources:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary={k:v for k,v in report.items() if k not in ('groups','audit_families_forced_to_train')};summary['cohort_sha256']=digest(out/'cohort.json')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
