"""Six consumed COCO development pairs; background-only intervention, no promotion."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT, digest
from background_intervention import make_variants, target_regions, region_delta

sys.path.insert(0, str(ROOT))
from reference_color_app.pipeline import analyze, algorithm_fingerprint
from reference_color_app.lut import apply_mapping
from spatial_render import SpatialEngine
from spatial_store import SpatialStore

EXPECTED = [(785,3553),(9483,19432),(872,1268),(8021,8690),(285,1993),(2149,2592)]


def main():
    import torch
    torch.set_num_threads(4)
    base=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v1'
    out.mkdir(exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    names=['background-intervention-protocol.md','background_intervention.py',
           'test_background_intervention.py','probe_background_intervention.py',
           'spatial_store.py','spatial_render.py','frame_transfer.py','aligned_spatial.py',
           'train_triplet_pilot.py','training_checks.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    rows=[r for r in json.loads((base/'development/jobs.json').read_text())['jobs'] if r['profile']=='identity']
    if [(r['target'],r['reference']) for r in rows]!=EXPECTED:raise ValueError('Development cohort differs')
    engine=SpatialEngine('multiref-v1')
    report=dict(scope='Consumed six COCO development pairs, background intervention; not independent quality or promotion',
                model_sha256=engine.checkpoint_sha,strength=.8,global_fingerprint=algorithm_fingerprint(),
                source_sha256={p.name:digest(p) for p in frozen.iterdir()},
                jobs_sha256=digest(base/'development/jobs.json'),manifest_sha256=digest(base/'manifest.json'),
                distance_metric='8-neighbor chessboard, capped at eight pixels',
                cases=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    def save(p,rgb):
        Image.fromarray(rgb).save(p)
        return dict(path=str(p.relative_to(out)),sha256=digest(p),pixel_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest())

    for row in rows:
        pair=f"{row['target']}-{row['reference']}";folder=out/pair;folder.mkdir()
        parents={}
        arrays={}
        for role in ['target','reference']:
            for kind in ['input','human-mask']:
                p=base/f"{row[role]}-{kind}.png";parents[str(p.relative_to(base))]=digest(p)
                with Image.open(p) as im:
                    if kind=='input':arrays[role]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                    else:arrays[role+'_mask']=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
        target,ref=arrays['target'],arrays['reference'];tm,rm=arrays['target_mask'],arrays['reference_mask']
        if row['group']=='none' and (tm.any() or rm.any()):raise ValueError('No-person control has people')
        if row['group']!='none' and (not tm.any() or not rm.any()):raise ValueError('Person cohort mask empty')
        input_file=save(folder/'target.png',target)
        save(folder/'target-mask.png',tm.astype(np.uint8)*255)
        save(folder/'reference-mask.png',rm.astype(np.uint8)*255)
        regions=target_regions(target,tm);baseline={};case=dict(pair=pair,group=row['group'],parents=parents,
            input_file=input_file,target_person_fraction=float(tm.mean()),reference_person_fraction=float(rm.mean()),variants=[])
        store=SpatialStore(folder/'store',engine)
        for name,rgb,info in make_variants(ref,rm):
            start=time.perf_counter();ref_file=save(folder/(name+'-reference.png'),rgb)
            # The duplicate runs through a fresh store to avoid reusing the prepared grid.
            active=SpatialStore(folder/'repeat-store',engine) if name=='identity_repeat' else store
            a,b=active.add_media(out/input_file['path']),active.add_media(out/ref_file['path'])
            mapping=active.analyze(a['id'],b['id']);decoded,_=active.decode(a)
            if not np.array_equal(decoded,target):raise ValueError('Unexpected target decode')
            adaptive=active.render(mapping['id'],target,.8)
            global_rgb=apply_mapping(target,analyze(out/input_file['path'],out/ref_file['path']),.8)
            variant=dict(name=name,reference=ref_file,intervention=info,
                         reference_delta={k:region_delta(rgb,ref,v) for k,v in target_regions(ref,rm).items()},
                         certified_aligned=mapping['diagnostics']['certified_aligned']['accepted'],methods={})
            for method,pixels in [('global',global_rgb),('adaptive',adaptive)]:
                if name=='identity':baseline[method]=pixels.copy()
                if name=='identity_repeat' and not np.array_equal(pixels,baseline[method]):raise ValueError('Repeat output differs')
                variant['methods'][method]=dict(file=save(folder/(name+'-'+method+'.png'),pixels),
                    regions={k:region_delta(pixels,baseline[method],mask) for k,mask in regions.items()})
            variant['seconds']=time.perf_counter()-start;case['variants'].append(variant)
        for p,h in parents.items():
            if digest(base/p)!=h:raise ValueError('Parent changed during run')
        report['cases'].append(case)
        (out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(pair=pair,variants=len(case['variants']),finished_pairs=len(report['cases']))),flush=True)
    if algorithm_fingerprint()!=report['global_fingerprint']:raise ValueError('Global code changed')
    if any(digest(Path(__file__).with_name(n))!=h for n,h in report['source_sha256'].items()):raise ValueError('Source changed')
    report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(finished=True,pairs=len(report['cases']),outputs=144)),flush=True)


if __name__=='__main__':main()
