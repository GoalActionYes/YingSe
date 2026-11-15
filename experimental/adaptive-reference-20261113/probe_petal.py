"""Frozen77-case official pretrained PETAL screen; predictions precede GT reads."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import torch
from independent_coco import ROOT,digest
from petal_model import PetalModel,tensor,rgb8
from spatial_render import SpatialEngine
from probe_region_histogram import score


def load(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))


def working(path,group):
    image=Image.fromarray(load(path))
    if group=='pst':image.thumbnail((640,640),Image.Resampling.BILINEAR)
    return np.array(image)


def jobs():
    pst=ROOT/'outputs/adaptive-reference-20261113/pst50';rows=[]
    for i in range(1,16):
        base=pst if i<=5 else pst/'development-dense-round2-v1/inputs'
        rows.append(dict(group='pst',index=i,target=base/f'content_709/in{i}.png',reference=base/f'paired_style/tar{i}.png',truth=base/f'paired_gt/gt{i}.png'))
    natural=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1';report=json.loads((natural/'results.json').read_text(encoding='utf-8'));assets={r['path']:r for r in report['assets']}
    for r in report['cases']:
        target=natural/r['outputs']['input']['path'];reference=natural/'srgb'/(r['triplet']['reference']+'.png')
        if digest(target)!=r['outputs']['input']['sha256'] or digest(reference)!=assets[r['triplet']['reference']]['sha256']:raise ValueError('Natural inputs changed')
        rows.append(dict(group='natural',index=r['index'],target=target,reference=reference,truth=natural/'srgb'/(r['triplet']['gt']+'.png'),truth_expected_sha256=assets[r['triplet']['gt']]['sha256']))
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco';j=json.loads((coco/'development/jobs.json').read_text(encoding='utf-8'))
    manifest=json.loads((coco/'manifest.json').read_text(encoding='utf-8'));lookup={r['id']:r for r in manifest['images']}
    if j['split']!='development' or len(j['jobs'])!=30:raise ValueError('Wrong consumed COCO split')
    for i,r in enumerate(j['jobs']):
        target=coco/f"{r['target']}-input.png"
        if digest(target)!=lookup[r['target']]['input_sha256']:raise ValueError('COCO input changed')
        rows.append(dict(group='coco',index=i,target=target,reference=coco/r['reference_file'],truth=coco/r['truth_file'],mask=coco/f"{r['target']}-human-mask.png",mask_expected_sha256=lookup[r['target']]['mask_sha256'],people=r['group'],profile=r['profile']))
    if [sum(r['group']==g for r in rows) for g in ['pst','natural','coco']]!=[15,32,30]:raise ValueError('Wrong77-case cohort')
    return rows


def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    out=ROOT/'outputs/adaptive-reference-20261113/petal-20261115-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_petal.py','petal_model.py','prepare_petal.py','petal-protocol.md','test_petal_model.py','spatial_render.py','frame_transfer.py','training_checks.py','train_triplet_pilot.py','probe_region_histogram.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    selected=jobs();model=PetalModel();baseline=SpatialEngine('multiref-v1')
    report=dict(scope='Consumed77-case methodscreen; not newheldout/human/video/product acceptance',candidate=model.provenance,baseline_checkpoint_sha256=baseline.checkpoint_sha,source_sha256={p.name:digest(p) for p in frozen.iterdir()},asset_manifest_sha256=digest(model.source/'manifest.json'),training=False,product_integrated=False,finished=False,cases=[],phase_events=dict(started_ns=time.time_ns()))
    report['model_state_before']=model.state_digest()
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Bridge uses publisher API and publisher PNG tensor loader, same restored model.
    bridges=[];rng=np.random.default_rng(42)
    for i,(w,h) in enumerate([(64,48),(73,51),(160,96)]):
        target=rng.integers(0,256,(h,w,3),dtype=np.uint8);reference=target[::-1,:,::-1].copy()
        a,b=out/f'bridge-{i}-source.png',out/f'bridge-{i}-reference.png';Image.fromarray(target).save(a);Image.fromarray(reference).save(b)
        ca,cb=model.api.load_image_tensor(str(a)).to(model.device),model.api.load_image_tensor(str(b)).to(model.device)
        inputdiff=max(float((tensor(target,model.device)-ca).abs().max()),float((tensor(reference,model.device)-cb).abs().max()))
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            direct=model.api.inference(model.style,model.transfer,ca,cb,guidance_scale=1.,style_input_size=512)
        image,raw,_=model.predict(target,reference);expected=direct[0].float().cpu().permute(1,2,0).numpy()
        difference=float(np.abs(expected-raw).max())
        if inputdiff!=0 or difference>1e-5 or not np.array_equal(image,rgb8(direct)):raise ValueError('Publisher bridge failed')
        bridges.append(dict(size=[w,h],maximum_input_difference=inputdiff,maximum_raw_difference=difference,rgb8_exact=True))
    report['publisher_bridges']=bridges
    for job in selected:
        try:
            target,reference=working(job['target'],job['group']),working(job['reference'],job['group'])
            candidate,raw,timing=model.predict(target,reference)
            start=time.perf_counter();current=baseline.prepare(target,reference).apply(target,1);baseline_seconds=time.perf_counter()-start
            prefix=f"{job['group']}-{job['index']}";files={}
            for name,value in [('baseline',current),('candidate',candidate)]:
                path=out/f'{prefix}-{name}.png';Image.fromarray(value).save(path);files[name]=dict(path=path.name,sha256=digest(path))
            rawfile=out/f'{prefix}-raw.npz';np.savez_compressed(rawfile,candidate=raw);files['raw']=dict(path=rawfile.name,sha256=digest(rawfile))
            info=dict(group=job['group'],index=job['index'],people=job.get('people'),profile=job.get('profile'),size=[target.shape[1],target.shape[0]],inputs={role:dict(path=str(job[role].relative_to(ROOT)),sha256=digest(job[role])) for role in ['target','reference']},files=files,candidate_timing=timing,baseline_seconds=baseline_seconds)
            report['cases'].append(info);print(json.dumps(dict(group=job['group'],index=job['index'],**timing)),flush=True)
        except Exception as exc:
            report['runtime_failure']=dict(group=job['group'],index=job['index'],type=type(exc).__name__,message=str(exc));(out/'runtime-failure.json').write_text(json.dumps(report,indent=2),encoding='utf-8');raise
    predictions=out/'predictions.json';predictions.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(predictions);report['phase_events']['predictions_written_ns']=time.time_ns()
    report['phase_events']['scoring_started_ns']=time.time_ns();report['all_predictions_before_truth_pixels']=True
    for row,job in zip(report['cases'],selected):
        target=working(job['target'],job['group']);truth=working(job['truth'],job['group'])
        if job['group']=='natural':truth=np.array(Image.fromarray(truth).resize((target.shape[1],target.shape[0]),Image.Resampling.BILINEAR))
        if truth.shape!=target.shape:raise ValueError('Truth and target working dimensions differ')
        if job.get('truth_expected_sha256') and digest(job['truth'])!=job['truth_expected_sha256']:raise ValueError('Truth changed')
        person=np.zeros(target.shape[:2],bool);boundary=person.copy()
        if job['group']=='coco':
            if digest(job['mask'])!=job['mask_expected_sha256']:raise ValueError('Mask changed')
            person=np.asarray(Image.open(job['mask']))>0
            edge=np.zeros(person.shape,np.uint8);edge[1:]|=person[1:]!=person[:-1];edge[:-1]|=person[1:]!=person[:-1];edge[:,1:]|=person[:,1:]!=person[:,:-1];edge[:,:-1]|=person[:,1:]!=person[:,:-1]
            boundary=np.asarray(Image.fromarray(edge*255).filter(ImageFilter.MaxFilter(7)))>0
        row['truth_sha256']=digest(job['truth']);row['scores']={}
        for name in ['baseline','candidate']:
            entry=row['files'][name];path=out/entry['path']
            if digest(path)!=entry['sha256']:raise ValueError('Prediction changed')
            output=load(path);s=score(output,truth,person,boundary)
            s['source_movement']=float(np.abs(output.astype(float)-target).mean())
            s['gradient_mae']=float((np.abs(np.diff(output.astype(float),axis=0)-np.diff(truth.astype(float),axis=0)).mean()+np.abs(np.diff(output.astype(float),axis=1)-np.diff(truth.astype(float),axis=1)).mean())/2)
            row['scores'][name]=s
    def aggregate(rows,epsilon):
        a=np.array([r['scores']['baseline']['mae'] for r in rows]);b=np.array([r['scores']['candidate']['mae'] for r in rows])
        return dict(count=len(rows),baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),improved=int((b<a-epsilon).sum()),regressed=int((b>a+epsilon).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()))
    by={g:[r for r in report['cases'] if r['group']==g] for g in ['pst','natural','coco']}
    regional=[r for r in by['coco'] if r['people']!='none' and r['profile'].startswith('region_')];controls=[r for r in by['coco'] if r not in regional]
    summary={g:aggregate(rows,.5 if g=='coco' else .1) for g,rows in by.items()};summary.update(regional=aggregate(regional,.5),controls=aggregate(controls,.5))
    summary['passed_gate']=all(summary[g]['candidate_mean']<=.95*summary[g]['baseline_mean'] and summary[g]['improved']>=n and summary[g]['severe']<=severe for g,n,severe in [('natural',22,2),('pst',10,1)]) and summary['regional']['candidate_mean']<=.9*summary['regional']['baseline_mean'] and summary['regional']['improved']>=6 and summary['controls']['regressed']<=2 and summary['coco']['severe']==0
    report['summary']=summary
    for group,count in [('pst',3),('natural',4),('coco',5)]:
        chosen=[(r,j) for r,j in zip(report['cases'],selected) if r['group']==group]
        for batch in range((len(chosen)+count-1)//count):
            sheet=Image.new('RGB',(5*220,count*200),'#202020');draw=ImageDraw.Draw(sheet)
            for y,(r,j) in enumerate(chosen[batch*count:(batch+1)*count]):
                panels=[('Target',working(j['target'],group)),('Reference',working(j['reference'],group)),('Known target',working(j['truth'],group))]+[(name,load(out/r['files'][name]['path'])) for name in ['baseline','candidate']]
                for x,(label,pixels) in enumerate(panels):
                    tile=Image.fromarray(pixels);tile.thumbnail((216,174),Image.Resampling.BILINEAR);sheet.paste(tile,(x*220,y*200+25));draw.text((x*220+3,y*200+3),str(r['index'])+' '+label,fill='white')
            sheet.save(out/f'{group}-sheet-{batch+1}.jpg',quality=94)
    report['model_state_after']=model.state_digest()
    if report['model_state_before']!=report['model_state_after']:raise ValueError('Frozen model state changed')
    if any(digest(Path(__file__).with_name(n))!=v for n,v in report['source_sha256'].items()):raise ValueError('Frozen source changed')
    for row in model.manifest['files']+model.manifest['weights']:
        if digest(model.source/row['path'])!=row['sha256']:raise ValueError('Author assets changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
