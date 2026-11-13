"""Metadata-only audit of same-content target variations, no images or scores."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';path=ROOT/'.tools/TST100K-index/triplet.json'
    if digest(path)!='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413':raise ValueError('Changed index')
    rows=json.loads(path.read_text(encoding='utf-8'));cohort_path=base/'cohort-v2/cohort.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'))
    known={Path(r['content']).stem for r in rows};audited=set(cohort['audit_families_forced_to_train'])
    def split(family):
        if family in audited:return 'train'
        value=int(hashlib.sha256(('yingse-tst-v1:'+family).encode()).hexdigest()[:8],16)%10
        return 'train' if value<8 else 'validation' if value==8 else 'reserved'
    result={}
    for name in ['train','validation']:
        selected=cohort['groups'][name];targets={r['content_family'] for r in selected};variants={f:set() for f in targets};current={f:set() for f in targets};unresolved=0
        for r in selected:current[r['content_family']].add(r['gt'])
        for r in rows:
            content=Path(r['content']).stem;reference=Path(r['reference']).stem.rsplit('_',1)[0]
            if content not in targets:continue
            if reference not in known:unresolved+=1;continue
            if split(reference)==name:variants[content].add(r['gt'])
        result[name]=dict(selected_triplets=len(selected),selected_unique_content=len(targets),selected_gt_path_counts=dict(Counter(len(v) for v in current.values())),
            available_gt_path_counts=dict(Counter(len(v) for v in variants.values())),targets_with_at_least_2_gt_paths=sum(len(v)>=2 for v in variants.values()),
            targets_with_at_least_4_gt_paths=sum(len(v)>=4 for v in variants.values()),unresolved_reference_rows_excluded=unresolved)
    report=dict(scope='Metadata only; distinct GT paths are not verified distinct visual styles; no causal shortcut claim',
        source_sha256=digest(Path(__file__)),index_sha256=digest(path),cohort_sha256=digest(cohort_path),known_reference_family_filter=True,groups=result)
    out=base/'context-data-audit-v2.json'
    if out.exists():raise ValueError('Audit output already exists')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
