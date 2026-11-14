"""ICC preparation and pretraining leakage screen, never compute quality scores."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from independent_coco import ROOT,digest
from triplet_color import to_srgb
from coverage_asset_plan import COHORT_SHA,validate_metadata
from coverage_duplicates import pixel_digest,perceptual_hash,near_crossings
from prepare_training_coverage import EXPECTED


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort=load(data/'cohort.json');manifest=load(data/'inputs-manifest.json');preflight=load(data/'metadata-preflight-v1/results.json');remote=load(data/'download-metadata.json');download=load(data/'download-v1/binding.json')
    paths={r[k] for p in ('train','validation') for r in cohort['groups'][p] for k in ('content','reference','gt')}
    if digest(data/'cohort.json')!=COHORT_SHA or not manifest['finished'] or manifest['cohort_sha256']!=COHORT_SHA or digest(data/'download-metadata.json')!=preflight['metadata_sha256'] or digest(data/'download-identity.json')!=preflight['identity_sha256'] or validate_metadata(remote,paths)!=manifest['total_bytes']:
        raise ValueError('Incomplete or changed byte preparation')
    if download['preflight_sha256']!=digest(data/'metadata-preflight-v1/results.json') or any(manifest[k]!=download[k] for k in download):raise ValueError('Download snapshot differs')
    protected={str(p.relative_to(data)):digest(p) for p in [data/'cohort.json',data/'inputs-manifest.json',data/'download-metadata.json',data/'download-identity.json',data/'metadata-preflight-v1/results.json',data/'download-v1/binding.json']}
    def verify_download_sources():
        for n,sha in download['source_sha256'].items():
            if digest(data/'download-v1/frozen-source'/n)!=sha or digest(Path(__file__).with_name(n))!=sha:raise ValueError('Download source changed')
    verify_download_sources()
    out=data/'srgb-v1';out.mkdir(exist_ok=True);work=out/'preparation-state';work.mkdir(exist_ok=True);frozen=work/'frozen-source';frozen.mkdir(exist_ok=True)
    names=['prepare_coverage_assets.py','coverage_duplicates.py','triplet_color.py','training-coverage-duplicates-protocol.md','training-coverage-duplicates-addendum.md'];sources={n:digest(Path(__file__).with_name(n)) for n in names}
    identity=dict(inputs=protected,source_sha256=sources)
    if (work/'identity.json').exists():
        if load(work/'identity.json')!=identity:raise ValueError('Preparation resume identity changed')
    else:
        for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
        (work/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    if any(digest(frozen/n)!=sha for n,sha in sources.items()):raise ValueError('Preparation frozen source changed')
    previous={};old_sources={};old_registry_paths={};held={}
    for name in ('cohort-v1','cohort-v2','cohort-multiref-v1'):
        old=base/name;old_cohort=load(old/'cohort.json');prep_path=old/'srgb-v1/manifest.json';prep=load(prep_path)
        if digest(old/'cohort.json')!=EXPECTED[name] or prep['cohort_sha256']!=EXPECTED[name]:raise ValueError('Old development metadata changed')
        old_sources[name]=digest(prep_path);old_registry_paths[name]=prep_path;validation={r[k] for r in old_cohort['groups']['validation'] for k in ('content','reference','gt')}
        for entry in prep['files']:
            registered=dict(entry,local_path=old/entry['prepared_path'])
            previous.setdefault(entry['original_path'],registered)
            if entry['original_path'] in validation:held.setdefault(entry['original_path'],[]).append(registered)
        if not validation<={r['original_path'] for r in prep['files']}:raise ValueError('Incomplete consumed validation registry')
    mapping={'reserved-v2':'cohort-v1','reserved-expanded-v1':'cohort-v2','reserved-multiref-v1':'cohort-multiref-reserved-v1'}
    for name,cohort_name in mapping.items():
        path=base/name/'results.json';record=load(path);old_sources[name]=digest(path);old_registry_paths[name]=path
        old_cohort_path=base/cohort_name/'cohort.json';old_cohort=load(old_cohort_path)
        expected_paths={r[k] for r in old_cohort['groups']['reserved'] for k in ('content','reference','gt')}
        if digest(old_cohort_path)!=EXPECTED[cohort_name] or record['cohort_sha256']!=EXPECTED[cohort_name] or len(record['cases'])!=32 or len(record['assets'])!=len(expected_paths) or {r['path'] for r in record['assets']}!=expected_paths:raise ValueError('Incomplete consumed reserved assets')
        for entry in record['assets']:
            held.setdefault(entry['path'],[]).append(dict(entry,original_path=entry['path'],local_path=base/name/entry['prepared_path']))
    # Freeze old registry identities before any new pixel decoding.
    old_path=work/'old-registry.json'
    if old_path.exists() and load(old_path)!=old_sources:raise ValueError('Consumed asset registry changed')
    old_path.write_text(json.dumps(old_sources,indent=2),encoding='utf-8')
    entries={v['path']:v for v in manifest['files']};metadata={v['path']:v for v in remote}
    if len(entries)!=len(manifest['files']) or set(entries)!=paths:raise ValueError('Downloaded asset set differs')
    def prepare(relative):
        entry=entries[relative];item=metadata[relative];original=data/'inputs'/relative;prepared=out/(relative+'.png');state=work/(hashlib.sha256(relative.encode()).hexdigest()+'.json')
        sha=digest(original)
        if sha!=entry['sha256'] or original.stat().st_size!=item['size']:raise ValueError('Original bytes changed')
        if 'lfs' in item:
            if sha!=item['lfs']['oid']:raise ValueError('Original LFS changed')
        else:
            raw=original.read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['oid']:raise ValueError('Original Git identity changed')
        if state.exists():
            record=load(state)
            if record['original_path']!=relative or record['original_sha256']!=sha or digest(prepared)!=record['sha256']:raise ValueError('Prepared resume identity changed')
            with Image.open(prepared) as im:
                if im.mode!='RGB' or not im.info.get('icc_profile') or min(im.size)<16:raise ValueError('Invalid resumed RGB8')
                rgb=np.array(im)
                if list(im.size)!=record['size'] or pixel_digest(rgb)!=record['pixel_sha256'] or perceptual_hash(rgb)!=record['perceptual_hash']:raise ValueError('Resumed pixels/fingerprint changed')
        else:
            prepared.parent.mkdir(parents=True,exist_ok=True);cached=previous.get(relative)
            if cached and cached['original_sha256']==sha:
                if digest(cached['local_path'])!=cached['sha256']:raise ValueError('Reusable prepared bytes changed')
                shutil.copyfile(cached['local_path'],prepared);info=cached['preparation'];reused=True
            else:
                with Image.open(original) as im:
                    if min(im.size)<16:raise ValueError('Unsupported source dimensions')
                    image,info=to_srgb(im)
                image.save(prepared,icc_profile=image.info['icc_profile']);reused=False
                with Image.open(prepared) as check:
                    if not np.array_equal(np.array(image),np.array(check)):raise ValueError('Prepared pixel roundtrip differs')
            with Image.open(prepared) as im:
                if im.mode!='RGB' or not im.info.get('icc_profile') or min(im.size)<16:raise ValueError('Invalid prepared RGB8')
                rgb=np.array(im)
            record=dict(original_path=relative,original_sha256=sha,prepared_path=str(prepared.relative_to(data)),sha256=digest(prepared),pixel_sha256=pixel_digest(rgb),size=list(im.size),perceptual_hash=perceptual_hash(rgb),preparation=info,reused_previous_prepared=reused)
            if cached and cached['original_sha256']==sha and record['pixel_sha256']!=cached['pixel_sha256']:raise ValueError('Reusable pixel identity differs')
            state.write_text(json.dumps(record,indent=2),encoding='utf-8')
        return record
    print('preparing RGB8 assets',len(paths),flush=True);done=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(prepare,p) for p in sorted(paths)]
        for future in as_completed(jobs):
            done.append(future.result())
            if len(done)%200==0:print('prepared',len(done),'/',len(paths),flush=True)
    done.sort(key=lambda r:r['original_path']);lookup={r['original_path']:r for r in done};invalid=[]
    for part in ('train','validation'):
        for r in cohort['groups'][part]:
            if lookup[r['content']]['size']!=lookup[r['gt']]['size']:invalid.append(dict(part=part,index=r['index']))
    held_records=[]
    for relative,registered in sorted(held.items()):
        hashes=set()
        for item in registered:
            if digest(item['local_path'])!=item['sha256']:raise ValueError('Consumed prepared bytes changed')
            with Image.open(item['local_path']) as im:rgb=np.array(im.convert('RGB'))
            pixel=pixel_digest(rgb)
            if pixel!=item['pixel_sha256']:raise ValueError('Consumed pixel identity changed')
            hashes.add(pixel)
        if len(hashes)!=1:raise ValueError('Consumed copies disagree in pixels')
        held_records.append(dict(path=relative,local_path=str(registered[0]['local_path'].relative_to(ROOT)),pixel_sha256=pixel,original_sha256=registered[0]['original_sha256'],hash=perceptual_hash(rgb)))
    train_paths={r[k] for r in cohort['groups']['train'] for k in ('content','reference','gt')};train=[dict(path=r['original_path'],hash=r['perceptual_hash'],pixel_sha256=r['pixel_sha256'],original_sha256=r['original_sha256']) for r in done if r['original_path'] in train_paths]
    exact=[dict(train=a['path'],heldout=b['path']) for a in train for b in held_records if a['pixel_sha256']==b['pixel_sha256'] or a['original_sha256']==b['original_sha256']]
    near=near_crossings(train,held_records);panels=out/'duplicate-panels';panels.mkdir(exist_ok=True);held_lookup={r['path']:ROOT/r['local_path'] for r in held_records}
    for i,flag in enumerate(near):
        panel=Image.new('RGB',(960,360),'#222222');draw=ImageDraw.Draw(panel)
        for column,(label,path) in enumerate([(flag['train'],data/lookup[flag['train']]['prepared_path']),(flag['heldout'],held_lookup[flag['heldout']])]):
            with Image.open(path) as im:thumb=ImageOps.contain(im.convert('RGB'),(472,320))
            panel.paste(thumb,(column*480,35));draw.text((column*480+5,8),label,fill='white')
        panel.save(panels/f'{i:04d}-hamming-{flag["distance"]}.jpg',quality=90)
    verify_download_sources()
    if any(digest(data/n)!=sha for n,sha in protected.items()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in sources.items()) or any(digest(old_registry_paths[n])!=sha for n,sha in old_sources.items()):raise ValueError('Preparation inputs/source changed midrun')
    report=dict(cohort_sha256=COHORT_SHA,inputs_manifest_sha256=protected['inputs-manifest.json'],source_sha256=sources,old_registry_sha256=old_sources,files=done,converted_files=len(done),heldout=held_records,training_unique_assets=len(train),heldout_unique_assets=len(held_records),invalid_pairs=invalid,cross_split_exact_duplicates=exact,cross_split_near_duplicates=near,passed=not(invalid or exact or near),training_started=False,scope='ICC RGB8/exact identity/limited63bit DCT duplicate screen; any flags require pretraining visual review, not exhaustive scene or pretraining isolation')
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(converted_files=len(done),heldout_unique_assets=len(held_records),training_unique_assets=len(train),invalid_pairs=len(invalid),exact_crossings=len(exact),near_crossings=len(near),passed=report['passed'],training_started=False)),flush=True)


if __name__=='__main__':main()
