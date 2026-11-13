"""Freeze same-content multiple-reference development groups from metadata."""
import hashlib
import json
from pathlib import Path
from independent_coco import ROOT,digest
from prepare_triplet_cohort import family


def select_grouped(rows,originals,group_count,styles,partition,audited):
    if type(group_count)!=int or group_count<1 or type(styles)!=int or styles<2 or partition not in ('train','validation'):raise ValueError('Invalid grouped cohort request')
    audited=set(audited);known={Path(r['content']).stem for r in rows}
    def split(f):
        if f in audited:return 'train'
        value=int(hashlib.sha256(('yingse-tst-v1:'+f).encode()).hexdigest()[:8],16)%10
        return 'train' if value<8 else 'validation' if value==8 else 'reserved'
    targets={r['content_family'] for r in originals};pool={f:[] for f in targets}
    if len(targets)!=len(originals):raise ValueError('Expected original unique-content cohort')
    ordered=sorted(enumerate(rows),key=lambda pair:hashlib.sha256(json.dumps(pair[1],sort_keys=True).encode()).digest())
    for index,row in ordered:
        a,b=Path(row['content']).stem,family(row['reference'])
        if a not in targets or b not in known or a==b or split(a)!=partition or split(b)!=partition:continue
        if family(row['gt'])!=a:raise ValueError('Unverified GT family')
        pool[a].append(dict(index=index,**row,content_family=a,reference_family=b))
    eligible=[]
    for original in originals:
        f=original['content_family'];choices=[dict(original)];used_gt={original['gt']};used_ref={original['reference']}
        if not any(r['index']==original['index'] for r in pool[f]):raise ValueError('Original pair is not eligible')
        for row in pool[f]:
            if row['gt'] in used_gt or row['reference'] in used_ref:continue
            choices.append(row);used_gt.add(row['gt']);used_ref.add(row['reference'])
            if len(choices)==styles:break
        if len(choices)==styles:eligible.append(choices)
    if len(eligible)<group_count:raise ValueError('Not enough complete multi-reference groups; do not shrink silently')
    selected=eligible[:group_count]
    result=[dict(group[i],group_id=group[i]['content_family'],style_index=i) for i in range(styles) for group in selected]
    return result,dict(eligible_groups=len(eligible),selected_groups=group_count,styles_per_group=styles)


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';index=ROOT/'.tools/TST100K-index/triplet.json';source=base/'cohort-v2/cohort.json'
    if digest(index)!='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413':raise ValueError('Changed index')
    old=json.loads(source.read_text(encoding='utf-8'));rows=json.loads(index.read_text(encoding='utf-8'))
    if digest(source)!='988889a59ec8c6591073a9d84f3cc20955d442e20cddf003f4dfcb622198862f':raise ValueError('Changed source cohort')
    train,ts=select_grouped(rows,old['groups']['train'],256,4,'train',old['audit_families_forced_to_train'])
    validation,vs=select_grouped(rows,old['groups']['validation'],32,2,'validation',old['audit_families_forced_to_train'])
    families=lambda data:{r[k] for r in data for k in ['content_family','reference_family']}
    if families(train)&families(validation):raise ValueError('Cross-split family leakage')
    consumed=set()
    for name in ['cohort-v1','cohort-v2']:
        past=json.loads((base/name/'cohort.json').read_text(encoding='utf-8'));consumed.update(families(past['groups']['reserved']))
    if (families(train)|families(validation))&consumed:raise ValueError('Consumed evaluation family in development')
    report=dict(dataset=old['dataset'],revision=old['revision'],index_sha256=digest(index),source_cohort_sha256=digest(source),source_sha256=digest(Path(__file__)),
        groups=dict(train=train,validation=validation,reserved=[]),selected_counts=dict(train=len(train),validation=len(validation),reserved=0),
        target_groups=dict(train=ts,validation=vs),audit_families_forced_to_train=old['audit_families_forced_to_train'],
        scope='Development-only same-content variation; no new independent set allocated; no images read',
        limitation='Distinct GT paths are not yet verified distinct visual styles; family is not subject/near-duplicate isolation',
        disjoint_photo_families=True,consumed_evaluation_families_excluded=True,training_started=False)
    out=base/'cohort-multiref-v1';out.mkdir(exist_ok=False)
    (out/'cohort.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'multiref-data-protocol.md').write_bytes(Path(__file__).with_name('multiref-data-protocol.md').read_bytes())
    print(json.dumps({k:report[k] for k in ['selected_counts','target_groups','consumed_evaluation_families_excluded']}))


if __name__=='__main__':main()
