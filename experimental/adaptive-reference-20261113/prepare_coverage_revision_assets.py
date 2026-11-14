"""Prepare v2 additions and recheck every training/consumed-heldout crossing."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from independent_coco import ROOT,digest
from triplet_color import to_srgb
from coverage_duplicates import pixel_digest,perceptual_hash,near_crossings
from coverage_asset_plan import validate_metadata


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';v1=base/'cohort-training-coverage-v1';data=base/'cohort-training-coverage-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');audit=load(data/'alias-evidence-audit-v1/results.json');added=load(data/'additions-v1/results.json');old=load(v1/'srgb-v1/manifest.json')
    if digest(data/'cohort.json')!='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077' or not audit['passed'] or not added['finished'] or audit['cohort_sha256']!=digest(data/'cohort.json') or added['cohort_sha256']!=digest(data/'cohort.json'):raise ValueError('Revision/evidence/additions not accepted')
    for n,sha in cohort['evidence_sha256'].items():
        if digest(v1/n)!=sha:raise ValueError('Alias evidence changed')
    paths={r[k] for part in ('train','validation') for r in cohort['groups'][part] for k in ('content','reference','gt')};remote={r['path']:r for r in load(data/'download-metadata.json')};validate_metadata(list(remote.values()),paths)
    sources={n:digest(Path(__file__).with_name(n)) for n in ['prepare_coverage_revision_assets.py','coverage_duplicates.py','triplet_color.py','coverage-alias-revision-protocol.md']}
    protected={str(p.relative_to(base)):digest(p) for p in [data/'cohort.json',data/'alias-evidence-audit-v1/results.json',data/'additions-v1/results.json',data/'download-metadata.json',v1/'srgb-v1/manifest.json']}
    out=data/'srgb-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for n in sources:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (out/'binding.json').write_text(json.dumps(dict(source_sha256=sources,protected_inputs_sha256=protected),indent=2),encoding='utf-8')
    previous={r['original_path']:r for r in old['files']};addition={r['path']:r for r in added['files']};done=[]
    for i,name in enumerate(sorted(paths),1):
        if name in previous:
            record=dict(previous[name]);p=(v1/record['prepared_path']).resolve()
            if digest(p)!=record['sha256'] or record['original_sha256']!=remote[name].get('lfs',{}).get('oid',record['original_sha256']):raise ValueError('Reusable identity differs')
            record['prepared_path']=os.path.relpath(p,data);record['reused_v1']=True
        else:
            raw=data/'inputs'/name;entry=addition[name]
            if digest(raw)!=entry['sha256'] or raw.stat().st_size!=remote[name]['size']:raise ValueError('Addition original changed')
            with Image.open(raw) as im:image,info=to_srgb(im)
            p=out/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True);image.save(p,icc_profile=image.info['icc_profile'])
            with Image.open(p) as im:
                rgb=np.array(im)
                if im.mode!='RGB' or not im.info.get('icc_profile') or not np.array_equal(rgb,np.array(image)):raise ValueError('RGB8 roundtrip failed')
            record=dict(original_path=name,original_sha256=entry['sha256'],prepared_path=str(p.relative_to(data)),sha256=digest(p),pixel_sha256=pixel_digest(rgb),size=list(image.size),perceptual_hash=perceptual_hash(rgb),preparation=info,reused_v1=False)
        if not p.resolve().is_relative_to(base.resolve()):raise ValueError('Shared asset escaped dataset root')
        with Image.open(p) as im:
            if im.mode!='RGB' or min(im.size)<16 or not im.info.get('icc_profile'):raise ValueError('Unsupported prepared image')
            rgb=np.array(im)
            if list(im.size)!=record['size'] or pixel_digest(rgb)!=record['pixel_sha256'] or perceptual_hash(rgb)!=record['perceptual_hash']:raise ValueError('Prepared pixel/fingerprint changed')
        done.append(record)
        if i%1000==0:print('verified v2 RGB assets',i,'/',len(paths),flush=True)
    lookup={r['original_path']:r for r in done};invalid=[]
    for part in ('train','validation'):
        for row in cohort['groups'][part]:
            if lookup[row['content']]['size']!=lookup[row['gt']]['size']:invalid.append(dict(part=part,index=row['index']))
    # Audit has just re-extracted all consumed fingerprints; independently recheck
    # the same immutable source pixels again before this revision's crossing scan.
    held=[]
    for r in old['heldout']:
        p=ROOT/r['local_path']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB'))
        if pixel_digest(rgb)!=r['pixel_sha256'] or perceptual_hash(rgb)!=r['hash']:raise ValueError('Consumed heldout pixels changed')
        held.append(dict(r))
    train_paths={r[k] for r in cohort['groups']['train'] for k in ('content','reference','gt')};train=[dict(path=r['original_path'],hash=r['perceptual_hash'],pixel_sha256=r['pixel_sha256'],original_sha256=r['original_sha256']) for r in done if r['original_path'] in train_paths]
    exact=[dict(train=a['path'],heldout=b['path']) for a in train for b in held if a['pixel_sha256']==b['pixel_sha256'] or a['original_sha256']==b['original_sha256']];near=near_crossings(train,held);panels=out/'duplicate-panels';panels.mkdir();held_lookup={r['path']:ROOT/r['local_path'] for r in held}
    for i,r in enumerate(near):
        canvas=Image.new('RGB',(960,360),'#222');draw=ImageDraw.Draw(canvas)
        for col,(label,p) in enumerate([(r['train'],data/lookup[r['train']]['prepared_path']),(r['heldout'],held_lookup[r['heldout']])]):
            with Image.open(p) as im:thumb=ImageOps.contain(im.convert('RGB'),(472,320))
            canvas.paste(thumb,(480*col,35));draw.text((480*col+5,8),label,fill='white')
        canvas.save(panels/f'{i:04d}-hamming-{r["distance"]}.jpg',quality=90)
    if any(digest(base/n)!=sha for n,sha in protected.items()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in sources.items()):raise ValueError('Revision/source changed midrun')
    result=dict(cohort_sha256=digest(data/'cohort.json'),source_sha256=sources,protected_inputs_sha256=protected,files=done,heldout=held,converted_files=len(done),training_unique_assets=len(train),heldout_unique_assets=len(held),invalid_pairs=invalid,cross_split_exact_duplicates=exact,cross_split_near_duplicates=near,passed=not(invalid or exact or near),training_started=False,storage_policy='Verified immutable shared v1 prepared paths within dataset root; only9new RGB8files created',scope='Full fixed v2 preparation and limited duplicate screen, not exhaustive scene/pretraining independence or quality acceptance')
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(files=len(done),new_rgb8=sum(not r['reused_v1'] for r in done),training_assets=len(train),heldout=len(held),invalid_pairs=len(invalid),exact=len(exact),near=len(near),passed=result['passed'],training_started=False)),flush=True)


if __name__=='__main__':main()
