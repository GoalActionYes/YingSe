"""Re-extract consumed fingerprints and bind reviewed panel cells to aliases."""
import io
import itertools
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from independent_coco import ROOT,digest
from coverage_duplicates import pixel_digest,perceptual_hash
from prepare_training_coverage import EXPECTED


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';v1=base/'cohort-training-coverage-v1';v2=base/'cohort-training-coverage-v2';assay=v1/'heldout-fingerprint-assay-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));record=load(assay/'results.json');revision=load(v2/'cohort.json');lookup={};registries={}
    if digest(v2/'cohort.json')!='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077':raise ValueError('Revision changed')
    for n in ('cohort-v1','cohort-v2','cohort-multiref-v1'):
        p=base/n/'srgb-v1/manifest.json';c=load(base/n/'cohort.json');m=load(p)
        if digest(base/n/'cohort.json')!=EXPECTED[n] or m['cohort_sha256']!=EXPECTED[n]:raise ValueError('Consumed source changed')
        wanted={r[k] for r in c['groups']['validation'] for k in ('content','reference','gt')};entries={r['original_path']:r for r in m['files']}
        if not wanted<=entries.keys():raise ValueError('Incomplete validation assets')
        registries[n]=digest(p)
        for name in wanted:lookup.setdefault(name,[]).append((base/n/entries[name]['prepared_path'],entries[name]))
    for n,cohort in {'reserved-v2':'cohort-v1','reserved-expanded-v1':'cohort-v2','reserved-multiref-v1':'cohort-multiref-reserved-v1'}.items():
        p=base/n/'results.json';m=load(p);c=load(base/cohort/'cohort.json');wanted={r[k] for r in c['groups']['reserved'] for k in ('content','reference','gt')}
        if m['cohort_sha256']!=EXPECTED[cohort] or {r['path'] for r in m['assets']}!=wanted or len(m['assets'])!=len(wanted):raise ValueError('Incomplete reserved assets')
        registries[n]=digest(p)
        for r in m['assets']:lookup.setdefault(r['path'],[]).append((base/n/r['prepared_path'],r))
    if registries!=record['registry_sha256'] or registries!=load(v1/'old-heldout-preflight-v1/results.json')['registry_sha256'] or digest(Path(__file__).with_name('coverage_duplicates.py'))!=record['helper_sha256']:raise ValueError('Assay source registry/helper changed')
    vectors=[];copies=0
    for name,entries in sorted(lookup.items()):
        pixels=set();hashes=set()
        for path,item in entries:
            if digest(path)!=item['sha256']:raise ValueError('Prepared byte identity changed')
            with Image.open(path) as im:a=np.array(im.convert('RGB'))
            value=pixel_digest(a)
            if value!=item['pixel_sha256']:raise ValueError('Pixel identity changed')
            pixels.add(value);hashes.add(perceptual_hash(a));copies+=1
        if len(pixels)!=1 or len(hashes)!=1:raise ValueError('Consumed copies disagree')
        family=Path(name).stem if name.startswith('content_images/') else Path(name).stem.rsplit('_',1)[0]
        vectors.append(dict(path=name,family=family,hash=next(iter(hashes))))
    if vectors!=load(assay/'private-fingerprints.json') or len(vectors)!=542:raise ValueError('Re-extracted fingerprints differ')
    flags=[];counts=dict(same_family_pairs=0,same_family_recovered=0,other_family_pairs=0,other_family_flagged=0)
    for a,b in itertools.combinations(vectors,2):
        same=a['family']==b['family'];distance=(a['hash']^b['hash']).bit_count();flag=distance<=6;counts['same_family_pairs' if same else 'other_family_pairs']+=1
        if flag:counts['same_family_recovered' if same else 'other_family_flagged']+=1
        if same and not flag or not same and flag:flags.append(dict(first=a['path'],second=b['path'],same_family=same,distance=distance))
    if flags!=load(assay/'private-flags.json') or counts!=record['counts']:raise ValueError('Flag ordering/counts differ')
    other=[v for v in flags if not v['same_family']];review=load(assay/'private-review.json');canvas=Image.new('RGB',(960,240*len(other)),'#222222');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(other):
        for col,key in enumerate(('first','second')):
            with Image.open(lookup[r[key]][0][0]) as im:thumb=ImageOps.contain(im.convert('RGB'),(472,210))
            canvas.paste(thumb,(col*480,i*240+27));draw.text((col*480+3,i*240+3),str(i)+' d='+str(r['distance'])+' '+r[key],fill='white')
    stream=io.BytesIO();canvas.save(stream,format='JPEG',quality=92);panel_sha=hashlib.sha256(stream.getvalue()).hexdigest()
    if panel_sha!=review['panel_sha256'] or panel_sha!=digest(assay/'other-family-flags.jpg') or sorted(v['index'] for v in review['decisions'])!=list(range(len(other))) or any(v['decision']!='same_capture_or_near_duplicate' for v in review['decisions']):raise ValueError('Manual indices/panel pixels not bound')
    for name,sha in revision['evidence_sha256'].items():
        if digest(v1/name)!=sha:raise ValueError('Revision evidence changed')
    for name,sha in revision['source_sha256'].items():
        if digest(v2/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Revision source changed')
    out=v2/'alias-evidence-audit-v1';out.mkdir(exist_ok=False)
    result=dict(source_sha256=digest(Path(__file__)),cohort_sha256=digest(v2/'cohort.json'),registry_sha256=registries,prepared_copies_reverified=copies,fingerprints_reextracted=len(vectors),full_flags_recomputed=len(flags),reviewed_other_family_pairs=len(other),review_panel_rebuilt_byte_exact=True,flags_sha256=digest(assay/'private-flags.json'),review_sha256=digest(assay/'private-review.json'),passed=True,training_started=False,scope='Re-extracted consumed heldout fingerprints and byte-exact reviewed panel binding; no new image quality scoring or exhaustive scene independence')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
