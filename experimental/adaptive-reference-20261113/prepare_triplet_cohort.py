"""Freeze academic triplets by source-photo families before image training."""
import hashlib
import argparse
import json
from pathlib import Path
from independent_coco import ROOT,digest


def family(path):return Path(path).stem.rsplit('_',1)[0]


def select_cohort(rows,limits,exclude_reserved_families=()):
    if set(limits)!={'train','validation','reserved'} or any(type(v)!=int or v<1 for v in limits.values()):raise ValueError('Invalid cohort sizes')
    known={Path(r['content']).stem for r in rows}
    if any(family(r['gt'])!=Path(r['content']).stem for r in rows):raise ValueError('Unverified source-family convention')
    audited={f for row in rows[:4] for f in [Path(row['content']).stem,family(row['reference'])]}
    def partition(name):
        if name in audited:return 'train'
        bucket=int(hashlib.sha256(('yingse-tst-v1:'+name).encode()).hexdigest()[:8],16)%10
        return 'train' if bucket<8 else 'validation' if bucket==8 else 'reserved'
    groups={k:[] for k in ['train','validation','reserved']};unresolved=0
    ordered=sorted(enumerate(rows),key=lambda item:hashlib.sha256(json.dumps(item[1],sort_keys=True).encode()).digest())
    used={k:set() for k in groups}
    for i,row in ordered:
        a,b=Path(row['content']).stem,family(row['reference'])
        if b not in known:unresolved+=1;continue
        part=partition(a)
        if partition(b)!=part or a==b or a in used[part]:continue
        used[part].add(a);groups[part].append(dict(index=i,**row,content_family=a,reference_family=b))
    available={k:len(v) for k,v in groups.items()}
    excluded=set(exclude_reserved_families)
    groups['reserved']=[r for r in groups['reserved'] if not ({r['content_family'],r['reference_family']}&excluded)]
    selected={k:groups[k][:limits[k]] for k in groups}
    if any(len(selected[k])!=limits[k] for k in selected):raise ValueError('Insufficient separated pairs')
    families={k:{f for row in value for f in [row['content_family'],row['reference_family']]} for k,value in selected.items()}
    if any(families[a]&families[b] for a,b in [('train','validation'),('train','reserved'),('validation','reserved')]):raise ValueError('Photo-family leakage')
    return selected,dict(audited=sorted(audited),unresolved=unresolved,available=available,reserved_eligible=len(groups['reserved']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    index=ROOT/'.tools/TST100K-index/triplet.json';rows=json.loads(index.read_text(encoding='utf-8'))
    if digest(index)!='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413':raise ValueError('Index changed')
    limits=dict(train=512 if args.expanded else 128,validation=64 if args.expanded else 32,reserved=32)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';excluded=set();previous=None
    if args.expanded:
        path=base/'cohort-v1/cohort.json'
        if digest(path)!='0592b2a06eedc111d35358d9b3542b081b61d49013e302b65e5f041b2527db6e':raise ValueError('Changed previous cohort')
        previous=json.loads(path.read_text(encoding='utf-8'))
        excluded={r[k] for r in previous['groups']['reserved'] for k in ['content_family','reference_family']}
    selected,stats=select_cohort(rows,limits,excluded)
    if previous and any(selected[p][:len(previous['groups'][p])]!=previous['groups'][p] for p in ['train','validation']):raise ValueError('Original development prefix changed')
    out=base/('cohort-v2' if args.expanded else 'cohort-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(dataset='ToneStyle/TST100K',revision='63bf13833b75407c862d608c3d81281d0a35b61f',index_sha256=digest(index),
        source_sha256=digest(Path(__file__)),purpose='Academic research pilot only, no product training/release',
        family_rule='Content stem; style stem before final underscore. Verified every GT family against author content path, skip unresolved references. Never infer triplet pairing.',
        limitation='Photo-family split is not a subject/scene-group annotation; related captures may still share subjects',
        audit_indices=[0,1,2,3],audit_families_forced_to_train=stats['audited'],unresolved_reference_rows_excluded=stats['unresolved'],
        available_unique_content_counts=stats['available'],selected_counts=limits,groups=selected,
        excluded_prior_reserved_families=sorted(excluded),reserved_eligible_after_exclusion=stats['reserved_eligible'],
        disjoint_photo_families=True,reserved_images_not_downloaded=True,training_started=False)
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'README-source.md').write_bytes((ROOT/'.tools/TST100K-index/README.md').read_bytes())
    if args.expanded:(out/'expanded-triplet-protocol.md').write_bytes(Path(__file__).with_name('expanded-triplet-protocol.md').read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)


if __name__=='__main__':main()
