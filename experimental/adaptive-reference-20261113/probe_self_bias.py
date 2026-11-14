"""No-training fixed62-case internal identity-bias diagnostic."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine,checkpoint_for
from self_bias import cancel
from probe_region_histogram import score


def load(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))


def main():
    torch.set_num_threads(4)
    natural=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1'
    nm=json.loads((natural/'results.json').read_text(encoding='utf-8'));assets={r['path']:r for r in nm['assets']}
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    cm=json.loads((coco/'development/jobs.json').read_text(encoding='utf-8'))
    cmf=json.loads((coco/'manifest.json').read_text(encoding='utf-8'));cassets={r['id']:r for r in cmf['images']}
    jobs=[]
    for row in nm['cases']:
        source=natural/row['outputs']['input']['path'];reference=natural/'srgb'/(row['triplet']['reference']+'.png')
        if digest(source)!=row['outputs']['input']['sha256'] or digest(reference)!=assets[row['triplet']['reference']]['sha256']:raise ValueError('Natural source changed')
        jobs.append(dict(group='natural',index=row['index'],target=source,reference=reference,truth=natural/'srgb'/(row['triplet']['gt']+'.png'),truth_expected_sha256=assets[row['triplet']['gt']]['sha256']))
    for i,row in enumerate(cm['jobs']):
        source=coco/f"{row['target']}-input.png"
        if digest(source)!=cassets[row['target']]['input_sha256']:raise ValueError('COCO source changed')
        jobs.append(dict(group='coco',index=i,target=source,reference=coco/row['reference_file'],truth=coco/row['truth_file'],profile=row['profile'],people=row['group'],mask=coco/f"{row['target']}-human-mask.png"))
    if len(jobs)!=62 or cm['split']!='development':raise ValueError('Wrong diagnostic cohort')
    out=ROOT/'outputs/adaptive-reference-20261113/self-bias-20261115-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_self_bias.py','self_bias.py','self-bias-protocol.md','test_self_bias.py','spatial_render.py','training_checks.py','train_triplet_pilot.py','probe_region_histogram.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    checkpoint,sha=checkpoint_for('multiref-v1');engine=SpatialEngine('multiref-v1')
    report=dict(scope='Consumed natural32 + controlledCOCO30 development; no independent or human acceptance',checkpoint_sha256=sha,source_sha256={p.name:digest(p) for p in frozen.iterdir()},dataset_sha256=dict(natural=digest(natural/'results.json'),coco=digest(coco/'development/jobs.json')),training=False,product_integrated=False,cases=[])
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for job in jobs:
        target,reference=load(job['target']),load(job['reference']);start=time.perf_counter()
        prepared=engine.prepare(target,reference);raw=prepared.float_output(target)
        baseline=np.rint(raw.clip(0,1)*255).astype(np.uint8)
        byte=prepared.apply(target,1)
        if not np.array_equal(baseline,byte):raise ValueError('Float RGB8 baseline differs from actual byte baseline')
        cross_seconds=time.perf_counter()-start;start=time.perf_counter()
        self_raw=engine.prepare(target,target).float_output(target);candidate=cancel(target,raw,self_raw)
        seconds=time.perf_counter()-start
        prefix=f"{job['group']}-{job['index']}";files={}
        self_image=np.rint(self_raw.clip(0,1)*255).astype(np.uint8)
        for name,image in [('baseline',baseline),('candidate',candidate),('self',self_image)]:
            path=out/f'{prefix}-{name}.png';Image.fromarray(image).save(path);files[name]=dict(path=path.name,sha256=digest(path))
        rawfile=out/f'{prefix}-raw.npz';np.savez_compressed(rawfile,cross=raw,self=self_raw);files['raw']=dict(path=rawfile.name,sha256=digest(rawfile))
        report['cases'].append(dict(group=job['group'],index=job['index'],profile=job.get('profile'),people=job.get('people'),inputs_sha256=dict(target=digest(job['target']),reference=digest(job['reference'])),files=files,cross_seconds=cross_seconds,self_additional_seconds=seconds,self_mae=float(np.abs(self_image.astype(float)-target).mean()),raw_self_bias_mae=float(np.abs(self_raw-target.astype(np.float32)/255).mean()*255),movement=float(np.abs(candidate.astype(float)-baseline).mean()),byte_baseline_exact=True))
        print(json.dumps(dict(group=job['group'],index=job['index'],self_mae=report['cases'][-1]['self_mae'])),flush=True)
    predictions=out/'predictions.json';predictions.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(predictions);report['all_predictions_before_truth_read']=True
    for row,job in zip(report['cases'],jobs):
        truth=load(job['truth']);target=load(job['target']);truth=np.array(Image.fromarray(truth).resize((target.shape[1],target.shape[0]),Image.Resampling.BILINEAR))
        if job.get('truth_expected_sha256') and digest(job['truth'])!=job['truth_expected_sha256']:raise ValueError('Natural truth changed')
        row['truth_sha256']=digest(job['truth']);row['scores']={}
        person=np.zeros(target.shape[:2],bool);boundary=person.copy()
        if job['group']=='coco':
            if digest(job['mask'])!=cassets[cm['jobs'][job['index']]['target']]['mask_sha256']:raise ValueError('Annotation changed')
            person=np.asarray(Image.open(job['mask']))>0
            edge=np.zeros(person.shape,np.uint8);edge[1:]|=person[1:]!=person[:-1];edge[:-1]|=person[1:]!=person[:-1];edge[:,1:]|=person[:,1:]!=person[:,:-1];edge[:,:-1]|=person[:,1:]!=person[:,:-1]
            boundary=np.asarray(Image.fromarray(edge*255).filter(ImageFilter.MaxFilter(7)))>0
        for name in ['baseline','candidate']:
            path=out/row['files'][name]['path']
            if digest(path)!=row['files'][name]['sha256']:raise ValueError('Prediction changed')
            image=load(path);row['scores'][name]=score(image,truth,person,boundary)
        for role in ['target','reference']:
            if digest(job[role])!=row['inputs_sha256'][role]:raise ValueError('Inputs changed')
    def aggregate(rows,epsilon):
        a=np.array([r['scores']['baseline']['mae'] for r in rows]);b=np.array([r['scores']['candidate']['mae'] for r in rows])
        return dict(count=len(rows),baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),improved=int((b<a-epsilon).sum()),regressed=int((b>a+epsilon).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()),self_mean=float(np.mean([r['self_mae'] for r in rows])),self_median=float(np.median([r['self_mae'] for r in rows])))
    nr=[r for r in report['cases'] if r['group']=='natural'];cr=[r for r in report['cases'] if r['group']=='coco'];rr=[r for r in cr if r['people']!='none' and r['profile'].startswith('region_')];negative=[r for r in cr if r not in rr]
    summary=dict(natural=aggregate(nr,.1),coco=aggregate(cr,.5),regional=aggregate(rr,.5),controls=aggregate(negative,.5))
    summary['passed_gate']=summary['natural']['candidate_mean']<=.95*summary['natural']['baseline_mean'] and summary['natural']['improved']>=22 and summary['natural']['severe']<=2 and summary['regional']['improved']>=6 and summary['regional']['candidate_mean']<=.9*summary['regional']['baseline_mean'] and summary['controls']['regressed']<=2 and summary['coco']['severe']==0
    report['summary']=summary
    # Fixed order: natural4 rows/sheet, then COCO5 rows/sheet. No score selection.
    for group,count in [('natural',4),('coco',5)]:
        selected=[(r,j) for r,j in zip(report['cases'],jobs) if r['group']==group]
        for batch in range((len(selected)+count-1)//count):
            sheet=Image.new('RGB',(5*220,count*200),'#202020');draw=ImageDraw.Draw(sheet)
            for y,(r,j) in enumerate(selected[batch*count:(batch+1)*count]):
                panels=[('Target',load(j['target'])),('Reference',load(j['reference'])),('Known target',load(j['truth']))]+[(name,load(out/r['files'][name]['path'])) for name in ['baseline','candidate']]
                for x,(label,image) in enumerate(panels):
                    tile=Image.fromarray(image);tile.thumbnail((216,174),Image.Resampling.BILINEAR);sheet.paste(tile,(x*220,y*200+25));draw.text((x*220+3,y*200+3),str(r['index'])+' '+label,fill='white')
            sheet.save(out/f'{group}-sheet-{batch+1}.jpg',quality=94)
    if digest(checkpoint)!=sha or any(digest(Path(__file__).with_name(n))!=v for n,v in report['source_sha256'].items()):raise ValueError('Weights/source changed')
    report['source_and_weights_unchanged']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
