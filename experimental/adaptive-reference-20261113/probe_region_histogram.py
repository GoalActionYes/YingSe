"""Fixed consumed-development diagnostic. Predictions precede all GT reads."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine,checkpoint_for
from region_histogram import predict


def load(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def score(rgb,truth,person,boundary):
    error=np.abs(rgb.astype(np.float64)-truth.astype(np.float64))
    def selected(mask):return float(error[mask].mean()) if mask.any() else None
    relation=None
    if person.any() and (~person).any():
        actual=rgb[person].mean(0)-rgb[~person].mean(0)
        desired=truth[person].mean(0)-truth[~person].mean(0)
        relation=float(np.abs(actual-desired).mean())
    return dict(mae=float(error.mean()),person_mae=selected(person),background_mae=selected(~person),relation_mae=relation,boundary_mae=selected(boundary),endpoint_fraction=float(((rgb==0)|(rgb==255)).mean()))


def main():
    torch.set_num_threads(4)
    data=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    jobfile=data/'development/jobs.json';jobsdata=json.loads(jobfile.read_text(encoding='utf-8'));jobs=jobsdata['jobs']
    if jobsdata['split']!='development' or len(jobs)!=30:raise ValueError('Wrong consumed development cohort')
    if digest(Path(__file__).with_name('controlled_styles.py'))!=jobsdata['controlled_style_source_sha256']:raise ValueError('Controlled producer changed')
    manifestfile=data/'manifest.json';manifest=json.loads(manifestfile.read_text(encoding='utf-8'))
    lookup={r['id']:r for r in manifest['images']}
    allowed={(p['target'],p['reference']) for p in manifest['pairs'] if p['split']=='development'}
    if len(allowed)!=6 or any((j['target'],j['reference']) not in allowed or j['split']!='development' for j in jobs):raise ValueError('Wrong pairs')
    masks=data/'development/automatic-masks';maskmeta=masks/'results.json'
    provenance=json.loads(maskmeta.read_text(encoding='utf-8'))['provenance']
    if provenance['sha256']!='fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70':raise ValueError('Wrong cached segmentation provenance')
    out=ROOT/'outputs/adaptive-reference-20261113/region-histogram-20261115-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_region_histogram.py','region_histogram.py','test_region_histogram.py','region-histogram-protocol.md','spatial_render.py','training_checks.py','train_triplet_pilot.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    source_hash={p.name:digest(p) for p in frozen.iterdir()}
    inputs={};prepared=[]
    for n,j in enumerate(jobs):
        paths=dict(target=data/f"{j['target']}-input.png",reference=data/j['reference_file'],target_mask=masks/f"{j['target']}-target.npy",reference_mask=masks/f"{j['reference']}-{j['profile']}.npy")
        if digest(paths['target'])!=lookup[j['target']]['input_sha256']:raise ValueError('Target changed')
        inputs[n]={k:digest(p) for k,p in paths.items()}
        prepared.append((load(paths['target']),load(paths['reference']),np.load(paths['target_mask'],allow_pickle=False),np.load(paths['reference_mask'],allow_pickle=False),paths))
    checkpoint,sha=checkpoint_for('multiref-v1');engine=SpatialEngine('multiref-v1')
    report=dict(scope='30 consumed controlled cases; not independent/natural/video acceptance',checkpoint_sha256=sha,source_sha256=source_hash,manifest_sha256=digest(manifestfile),jobs_sha256=digest(jobfile),segmentation=provenance,mask_metadata_sha256=digest(maskmeta),product_integrated=False,training=False,cases=[])
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for n,(j,values) in enumerate(zip(jobs,prepared)):
        target,reference,p,r,paths=values;start=time.perf_counter()
        baseline=engine.prepare(target,reference).apply(target,1);base_seconds=time.perf_counter()-start
        start=time.perf_counter();candidate,diagnostic=predict(engine,target,reference,p,r,baseline);torch.cuda.synchronize();seconds=time.perf_counter()-start
        outputs={}
        for name,rgb in [('baseline',baseline),('candidate',candidate)]:
            path=out/f'{n:02d}-{name}.png';Image.fromarray(rgb).save(path);outputs[name]=dict(file=path.name,sha256=digest(path))
        report['cases'].append(dict(index=n,job=j,inputs_sha256=inputs[n],outputs=outputs,diagnostic=diagnostic,baseline_seconds=base_seconds,candidate_additional_seconds=seconds,change_from_baseline=float(np.abs(candidate.astype(float)-baseline).mean())))
        print(json.dumps(dict(index=n,profile=j['profile'],applied=diagnostic['applied'])),flush=True)
    predictions=out/'predictions.json';predictions.write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['predictions_sha256']=digest(predictions);report['all_predictions_before_truth_read']=True
    sheets={}
    for row,values in zip(report['cases'],prepared):
        j=row['job'];target,reference,_,_,paths=values
        for role,path in paths.items():
            if digest(path)!=row['inputs_sha256'][role]:raise ValueError('Input changed during inference')
        truthpath=data/j['truth_file'];maskpath=data/f"{j['target']}-human-mask.png"
        if digest(maskpath)!=lookup[j['target']]['mask_sha256']:raise ValueError('Human annotation changed')
        truth=load(truthpath);person=np.array(Image.open(maskpath))>0
        edge=np.zeros(person.shape,np.uint8)
        edge[1:]|=(person[1:]!=person[:-1]);edge[:-1]|=(person[1:]!=person[:-1]);edge[:,1:]|=(person[:,1:]!=person[:,:-1]);edge[:,:-1]|=(person[:,1:]!=person[:,:-1])
        boundary=np.array(Image.fromarray(edge*255).filter(ImageFilter.MaxFilter(7)))>0
        row['truth_sha256']=digest(truthpath);row['human_mask_sha256']=digest(maskpath);row['scores']={}
        displayed=[('Target',target),('Reference',reference),('Known transform',truth)]
        for name,entry in row['outputs'].items():
            path=out/entry['file']
            if digest(path)!=entry['sha256']:raise ValueError('Prediction changed')
            rgb=load(path);row['scores'][name]=score(rgb,truth,person,boundary);displayed.append((name,rgb))
        key=(j['target'],j['reference']);sheet=sheets.setdefault(key,Image.new('RGB',(5*230,5*200),'#202020'));draw=ImageDraw.Draw(sheet)
        y=(row['index']%5)*200
        for col,(name,rgb) in enumerate(displayed):
            picture=Image.fromarray(rgb);picture.thumbnail((226,174),Image.Resampling.BILINEAR);sheet.paste(picture,(col*230,y+25));draw.text((col*230+3,y+3),j['profile']+' / '+name,fill='white')
    for (a,b),sheet in sheets.items():sheet.save(out/f'{a}-{b}-all-profiles.jpg',quality=94)
    def aggregate(rows):
        base=np.array([r['scores']['baseline']['mae'] for r in rows]);cand=np.array([r['scores']['candidate']['mae'] for r in rows])
        return dict(count=len(rows),baseline_mean=float(base.mean()),candidate_mean=float(cand.mean()),improved_gt_half=int((cand<base-.5).sum()),regressed_gt_half=int((cand>base+.5).sum()),severe=int(((cand>base+5)&(cand>base*1.25)).sum()),applied=sum(r['diagnostic']['applied'] for r in rows))
    regional=[r for r in report['cases'] if r['job']['group']!='none' and r['job']['profile'].startswith('region_')]
    negative=[r for r in report['cases'] if r not in regional]
    summary=dict(all=aggregate(report['cases']),regional=aggregate(regional),negative=aggregate(negative),by_profile={p:aggregate([r for r in report['cases'] if r['job']['profile']==p]) for p in sorted({j['profile'] for j in jobs})})
    summary['passed_development_gate']=summary['regional']['improved_gt_half']>=6 and summary['regional']['candidate_mean']<=.9*summary['regional']['baseline_mean'] and summary['negative']['regressed_gt_half']<=2 and summary['all']['severe']==0
    report['summary']=summary
    if digest(checkpoint)!=sha or any(digest(Path(__file__).with_name(n))!=v for n,v in source_hash.items()):raise ValueError('Source or weights changed')
    report['source_and_weights_unchanged']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
