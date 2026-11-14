"""Deterministic metadata-only training expansion; never read image pixels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from prepare_triplet_cohort import family


def expand_training(rows, old_train, heldout_cohorts, target_groups, audited):
    if type(target_groups) is not int or target_groups < 1:
        raise ValueError('Expected a positive target group count')
    known={}
    for row in rows:
        name=Path(row['content']).stem
        if name in known and known[name]!=row['content']:
            raise ValueError('Ambiguous content family paths')
        known[name]=row['content']
        if family(row['gt'])!=name:
            raise ValueError('Unverified GT family convention')
    audited=set(audited)
    def train(name):
        bucket=int(hashlib.sha256(('yingse-tst-v1:'+name).encode()).hexdigest()[:8],16)%10
        return name in audited or bucket<8
    excluded={r[k] for cohort in heldout_cohorts for part,part_rows in cohort['groups'].items()
              if part in ('validation','reserved') for r in part_rows
              for k in ('content_family','reference_family')}
    def eligible(row):
        a,b=Path(row['content']).stem,family(row['reference'])
        return b in known and a!=b and train(a) and train(b) and not ({a,b}&excluded)
    old=defaultdict(dict)
    for row in old_train:
        index=row['index'];style=row['style_index'];name=row['content_family']
        if type(index) is not int or not 0<=index<len(rows) or type(style) is not int or style not in range(4):
            raise ValueError('Invalid old group identity')
        if any(row[k]!=rows[index][k] for k in ('content','reference','gt')) or not eligible(row):
            raise ValueError('Old training pairing changed or intersects heldout sources')
        if name!=Path(row['content']).stem or row['group_id']!=name or row['reference_family']!=family(row['reference']):
            raise ValueError('Old family metadata differs')
        if style in old[name]:raise ValueError('Repeated old group style')
        old[name][style]=dict(row)
    if not old or target_groups<len(old):raise ValueError('Cannot shrink old training groups')
    for group in old.values():
        if set(group)!=set(range(4)) or any(len({r[k] for r in group.values()})!=4 for k in ('reference','gt')):
            raise ValueError('Old groups must have four distinct references/GT')
    pools=defaultdict(list);used=defaultdict(lambda:dict(reference=set(),gt=set()))
    ordered=sorted(enumerate(rows),key=lambda item:(hashlib.sha256(json.dumps(item[1],sort_keys=True).encode()).digest(),item[0]))
    for index,row in ordered:
        name=Path(row['content']).stem
        if name in old or not eligible(row) or len(pools[name])>=4:continue
        if row['reference'] in used[name]['reference'] or row['gt'] in used[name]['gt']:continue
        style=len(pools[name]);pools[name].append(dict(index=index,**row,content_family=name,reference_family=family(row['reference']),group_id=name,style_index=style))
        for k in ('reference','gt'):used[name][k].add(row[k])
    complete=[group for group in pools.values() if len(group)==4];needed=target_groups-len(old)
    if len(complete)<needed:raise ValueError('Insufficient complete new groups; counts not reduced')
    groups=[list(group[i] for i in range(4)) for group in old.values()]+complete[:needed]
    result=[dict(group[style]) for style in range(4) for group in groups]
    return result,dict(old_groups=len(old),added_groups=needed,target_groups=target_groups,triplets=len(result),
                       eligible_new_groups=len(complete),excluded_families=len(excluded),index_content_families=len(known),
                       no_image_pixels_read=True,photo_family_isolation_only=True)
