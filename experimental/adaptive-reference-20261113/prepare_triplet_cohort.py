"""Freeze academic triplets by source-photo families before image training."""
import hashlib
import json
from pathlib import Path
from independent_coco import ROOT,digest


def family(path):return Path(path).stem.rsplit('_',1)[0]


def main():
    index=ROOT/'.tools/TST100K-index/triplet.json';rows=json.loads(index.read_text(encoding='utf-8'))
    if digest(index)!='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413':raise ValueError('Index changed')
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
    limits=dict(train=128,validation=32,reserved=32)
    selected={k:groups[k][:limits[k]] for k in groups}
    if any(len(selected[k])!=limits[k] for k in selected):raise ValueError('Insufficient separated pairs')
    families={k:{f for row in value for f in [row['content_family'],row['reference_family']]} for k,value in selected.items()}
    if any(families[a]&families[b] for a,b in [('train','validation'),('train','reserved'),('validation','reserved')]):raise ValueError('Photo-family leakage')
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/cohort-v1';out.mkdir(parents=True,exist_ok=False)
    report=dict(dataset='ToneStyle/TST100K',revision='63bf13833b75407c862d608c3d81281d0a35b61f',index_sha256=digest(index),
        source_sha256=digest(Path(__file__)),purpose='Academic research pilot only, no product training/release',
        family_rule='Content stem; style stem before final underscore. Verified every GT family against author content path, skip unresolved references. Never infer triplet pairing.',
        limitation='Photo-family split is not a subject/scene-group annotation; related captures may still share subjects',
        audit_indices=[0,1,2,3],audit_families_forced_to_train=sorted(audited),unresolved_reference_rows_excluded=unresolved,
        available_unique_content_counts={k:len(v) for k,v in groups.items()},selected_counts=limits,groups=selected,
        disjoint_photo_families=True,reserved_images_not_downloaded=True,training_started=False)
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'README-source.md').write_bytes((ROOT/'.tools/TST100K-index/README.md').read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)


if __name__=='__main__':main()
