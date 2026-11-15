"""Fixed77 consumed development cases; frozen encoder, per-pair color solve only."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import torch
from independent_coco import ROOT,digest
from petal_model import PetalModel
from petal_encoder import encode
from petal_retouch import retouch,normalized,solve,select
from probe_petal import jobs,working,load
from probe_region_histogram import score


def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    prior_dir=ROOT/'outputs/adaptive-reference-20261113/petal-20261115-v1'
    prior_path=prior_dir/'results.json';prior=json.loads(prior_path.read_text(encoding='utf-8'))
    if not prior['finished'] or prior['baseline_checkpoint_sha256']!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Unexpected frozen baseline')
    selected=jobs()
    if len(prior['cases'])!=77:raise ValueError('Wrong historical case count')
    for row,job in zip(prior['cases'],selected):
        if (row['group'],row['index'])!=(job['group'],job['index']):raise ValueError('Historical ordering differs')
        for role in ['target','reference']:
            if digest(job[role])!=row['inputs'][role]['sha256'] or str(job[role].relative_to(ROOT))!=row['inputs'][role]['path']:raise ValueError('Historical input differs')
        if digest(prior_dir/row['files']['baseline']['path'])!=row['files']['baseline']['sha256']:raise ValueError('Historical baseline differs')
    out=ROOT/'outputs/adaptive-reference-20261113/petal-retouch-20261115-v1';out.mkdir(exist_ok=False)
    source_dir=Path(__file__).parent
    names=['probe_petal_retouch.py','petal_retouch.py','test_petal_retouch.py','petal-retouch-protocol.md','petal_model.py','petal_encoder.py','probe_petal.py','probe_region_histogram.py','spatial_render.py','independent_coco.py']
    sources={str((source_dir/n).relative_to(ROOT)):source_dir/n for n in names}
    sources['reference_color_app/adjustments.py']=ROOT/'reference_color_app/adjustments.py'
    frozen=out/'frozen-source';frozen.mkdir()
    for name,path in sources.items():
        dst=frozen/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
    report=dict(scope='Consumed77 development pairs; frozen PETAL encoder-guided bounded retouch, not independent quality acceptance',prior_results_sha256=digest(prior_path),baseline_checkpoint_sha256=prior['baseline_checkpoint_sha256'],source_sha256={n:digest(p) for n,p in sources.items()},finished=False,training=False,product_integrated=False,cases=[],phase_events=dict(started_ns=time.time_ns()))
    model=PetalModel();report['provenance']=model.provenance;report['model_state_before']=model.state_digest();report['asset_manifest_sha256']=digest(model.source/'manifest.json')
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for old,job in zip(prior['cases'],selected):
        baseline=load(prior_dir/old['files']['baseline']['path']);target=working(job['target'],job['group']);reference=working(job['reference'],job['group'])
        if baseline.shape!=target.shape:raise ValueError('Historical working size differs')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        initial=encode(model,[reference,baseline]).cpu().numpy()
        stencil=[]
        for axis in range(5):
            u=np.zeros(5);u[axis]=1
            stencil.extend(encode(model,[retouch(baseline,u),retouch(baseline,-u)]).cpu().numpy())
        stencil=np.stack(stencil);u=solve(np.concatenate([initial[1:2],stencil]),initial[0])
        options=[baseline]+[retouch(baseline,u*alpha) for alpha in [1.,.5,.25]]
        option_features=np.concatenate([encode(model,options[1:3]).cpu().numpy(),encode(model,[options[3],baseline]).cpu().numpy()])
        z=normalized(np.concatenate([initial[1:2],option_features[:3]]));ref=normalized(initial[:1])[0]
        distances=np.clip(1-z@ref,0,2);endpoints=np.array([((a==0)|(a==255)).mean() for a in options]);choice=select(distances,endpoints)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        prefix=f"{job['group']}-{job['index']}";files={}
        for name,pixels in [('baseline',baseline)]+[(f'option-{i}',v) for i,v in enumerate(options[1:],1)]+[('candidate',options[choice])]:
            path=out/f'{prefix}-{name}.png';Image.fromarray(pixels).save(path);files[name]=dict(path=path.name,sha256=digest(path))
        cache=out/f'{prefix}-features.npz';np.savez_compressed(cache,initial=initial,stencil=stencil,option_features=option_features,u=u,distances=distances,endpoints=endpoints)
        files['features']=dict(path=cache.name,sha256=digest(cache))
        report['cases'].append(dict(group=job['group'],index=job['index'],people=job.get('people'),profile=job.get('profile'),inputs=old['inputs'],prior_baseline_sha256=old['files']['baseline']['sha256'],files=files,u=u.tolist(),choice=choice,cosine_distances=distances.tolist(),endpoint_fractions=endpoints.tolist(),additional_analysis_seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(group=job['group'],index=job['index'],choice=choice,cosine_before=float(distances[0]),cosine_after=float(distances[choice]),seconds=seconds)),flush=True)
    predictions=out/'predictions.json';predictions.write_text(json.dumps(report,indent=2),encoding='utf-8');report['predictions_sha256']=digest(predictions);report['phase_events']['predictions_written_ns']=time.time_ns()
    report['phase_events']['scoring_started_ns']=time.time_ns();report['all_predictions_before_truth_pixels']=True
    for row,old,job in zip(report['cases'],prior['cases'],selected):
        target=working(job['target'],job['group']);truth=working(job['truth'],job['group'])
        if job['group']=='natural':truth=np.array(Image.fromarray(truth).resize((target.shape[1],target.shape[0]),Image.Resampling.BILINEAR))
        if truth.shape!=target.shape or digest(job['truth'])!=old['truth_sha256']:raise ValueError('Historical truth differs')
        person=np.zeros(target.shape[:2],bool);boundary=person.copy()
        if job['group']=='coco':
            if digest(job['mask'])!=job['mask_expected_sha256']:raise ValueError('Mask changed')
            person=np.asarray(Image.open(job['mask']))>0
            edge=np.zeros(person.shape,np.uint8);edge[1:]|=person[1:]!=person[:-1];edge[:-1]|=person[1:]!=person[:-1];edge[:,1:]|=person[:,1:]!=person[:,:-1];edge[:,:-1]|=person[:,1:]!=person[:,:-1]
            boundary=np.asarray(Image.fromarray(edge*255).filter(ImageFilter.MaxFilter(7)))>0
        row['truth_sha256']=digest(job['truth']);row['scores']={}
        for name in ['baseline','candidate']:
            path=out/row['files'][name]['path']
            if digest(path)!=row['files'][name]['sha256']:raise ValueError('Prediction changed')
            output=load(path);s=score(output,truth,person,boundary)
            s['source_movement']=float(np.abs(output.astype(float)-target).mean())
            s['gradient_mae']=float((np.abs(np.diff(output.astype(float),axis=0)-np.diff(truth.astype(float),axis=0)).mean()+np.abs(np.diff(output.astype(float),axis=1)-np.diff(truth.astype(float),axis=1)).mean())/2)
            row['scores'][name]=s
        if row['scores']['baseline']!=old['scores']['baseline']:raise ValueError('Historical baseline score differs')
        for role in ['target','reference']:
            if digest(job[role])!=row['inputs'][role]['sha256']:raise ValueError('Input changed during run')
    def aggregate(rows,epsilon):
        a=np.array([r['scores']['baseline']['mae'] for r in rows]);b=np.array([r['scores']['candidate']['mae'] for r in rows])
        return dict(count=len(rows),baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),improved=int((b<a-epsilon).sum()),regressed=int((b>a+epsilon).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()),selected=int(sum(r['choice']!=0 for r in rows)))
    by={g:[r for r in report['cases'] if r['group']==g] for g in ['pst','natural','coco']}
    regional=[r for r in by['coco'] if r['people']!='none' and r['profile'].startswith('region_')];controls=[r for r in by['coco'] if r not in regional]
    summary={g:aggregate(rows,.5 if g=='coco' else .1) for g,rows in by.items()};summary.update(regional=aggregate(regional,.5),controls=aggregate(controls,.5))
    summary['passed_gate']=all(summary[g]['candidate_mean']<=.95*summary[g]['baseline_mean'] and summary[g]['improved']>=n and summary[g]['severe']<=severe for g,n,severe in [('natural',22,2),('pst',10,1)]) and summary['regional']['candidate_mean']<=.9*summary['regional']['baseline_mean'] and summary['regional']['improved']>=6 and summary['controls']['regressed']<=2 and summary['coco']['severe']==0
    report['summary']=summary
    for group,count in [('pst',3),('natural',4),('coco',5)]:
        pairs=[(r,j) for r,j in zip(report['cases'],selected) if r['group']==group]
        for batch in range((len(pairs)+count-1)//count):
            sheet=Image.new('RGB',(5*220,count*200),'#202020');draw=ImageDraw.Draw(sheet)
            for y,(row,job) in enumerate(pairs[batch*count:(batch+1)*count]):
                panels=[('Target',working(job['target'],group)),('Reference',working(job['reference'],group)),('Known target',working(job['truth'],group))]+[(name,load(out/row['files'][name]['path'])) for name in ['baseline','candidate']]
                for x,(label,pixels) in enumerate(panels):
                    tile=Image.fromarray(pixels);tile.thumbnail((216,174),Image.Resampling.BILINEAR);sheet.paste(tile,(x*220,y*200+25));draw.text((x*220+3,y*200+3),str(row['index'])+' '+label,fill='white')
            sheet.save(out/f'{group}-sheet-{batch+1}.jpg',quality=94)
    report['model_state_after']=model.state_digest()
    if report['model_state_before']!=report['model_state_after'] or any(m.training or any(p.requires_grad for p in m.parameters()) for m in [model.style,model.transfer]):raise ValueError('Model state changed')
    if digest(prior_path)!=report['prior_results_sha256'] or any(digest(p)!=report['source_sha256'][n] for n,p in sources.items()):raise ValueError('Frozen evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
