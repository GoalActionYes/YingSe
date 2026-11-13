"""Freeze the next reserved metadata, retiring both consumed cohorts."""
import json
from pathlib import Path
from independent_coco import ROOT,digest
from prepare_triplet_cohort import select_cohort


def select_fresh_reserved(rows,consumed,development,count=32):
    excluded=set()
    for cohort in consumed:
        for row in cohort['groups']['reserved']:excluded.update(row[k] for k in ['content_family','reference_family'])
    for part in ['train','validation']:
        for row in development['groups'][part]:excluded.update(row[k] for k in ['content_family','reference_family'])
    selected,stats=select_cohort(rows,dict(train=1,validation=1,reserved=count),excluded)
    stats['excluded_families']=sorted(excluded)
    return selected['reserved'],stats


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';index=ROOT/'.tools/TST100K-index/triplet.json'
    expected={'cohort-v1':'0592b2a06eedc111d35358d9b3542b081b61d49013e302b65e5f041b2527db6e',
        'cohort-v2':'988889a59ec8c6591073a9d84f3cc20955d442e20cddf003f4dfcb622198862f',
        'cohort-multiref-v1':'124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d'}
    cohorts={}
    for name,sha in expected.items():
        path=base/name/'cohort.json'
        if digest(path)!=sha:raise ValueError('Changed source cohort')
        cohorts[name]=json.loads(path.read_text(encoding='utf-8'))
    if digest(index)!='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413':raise ValueError('Changed index')
    reserved,stats=select_fresh_reserved(json.loads(index.read_text(encoding='utf-8')),[cohorts['cohort-v1'],cohorts['cohort-v2']],cohorts['cohort-multiref-v1'])
    report=dict(dataset=cohorts['cohort-v2']['dataset'],revision=cohorts['cohort-v2']['revision'],index_sha256=digest(index),source_cohort_sha256=expected,
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['prepare_fresh_reserved.py','prepare_triplet_cohort.py','multiref-independent-protocol.md']},
        groups=dict(reserved=reserved),excluded_families=stats['excluded_families'],reserved_eligible=stats['reserved_eligible'],
        scope='New reserved metadata only, no images read; source-family split does not establish subject or pretrained independence')
    out=base/'cohort-multiref-reserved-v1';out.mkdir(exist_ok=False)
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'multiref-independent-protocol.md').write_bytes(Path(__file__).with_name('multiref-independent-protocol.md').read_bytes())
    print(json.dumps(dict(reserved=len(reserved),excluded_families=len(report['excluded_families']),eligible=report['reserved_eligible'])))


if __name__=='__main__':main()
