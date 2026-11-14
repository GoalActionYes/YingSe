"""Audit radius calibration, complete matched trial and fixed-weight raw ablation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from radius_context import radius_forward,calibrated_radius
from paired_training import grouped_batches
from pairwise_gate import require_completed_control,response_development_gate
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';candidate=base/'radius-context-v1';control=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(candidate/'results.json');c=load(control/'results.json');cal_path=base/'context-radius-calibration-v1/results.json';cal=load(cal_path)
    require_completed_control(c,r['config'],c['source_sha256']);require_completed_control(r,c['config'],r['source_sha256'])
    snapshot=dict(results_sha256=digest(control/'results.json'),scores_sha256=digest(control/f'epoch-{c["best_epoch"]}/scores.json'),selected_sha256=digest(control/f'epoch-{c["best_epoch"]}.pth'),initial_sha256=digest(control/'epoch-0.pth'))
    if snapshot!=r['control_snapshot'] or digest(cal_path)!=r['calibration_results_sha256'] or not cal['finished'] or r['radius']!=cal['radius']:raise ValueError('Changed trial/calibration/control')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);lookup={v['original_path']:v for v in prep['files']};rows=cohort['groups']['validation'];train=cohort['groups']['train'];checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if any(record['cohort_sha256']!=digest(cohort_path) or record['prepared_manifest_sha256']!=digest(prep_path) for record in [r,c,cal]) or cal['checkpoint_sha256']!=digest(checkpoint) or r['start_checkpoint_sha256']!=digest(checkpoint):raise ValueError('Changed data/base')
    for entry in prep['files']:
        if digest(data/entry['prepared_path'])!=entry['sha256']:raise ValueError('Changed prepared input')
    expected=[dict(index=v['index'],role=role,path=v[role],sha256=lookup[v[role]]['sha256']) for v in train for role in ['content','reference']]
    if len(cal['inputs'])!=2048 or any({k:v[k] for k in x}!=x for x,v in zip(expected,cal['inputs'])) or calibrated_radius([v['norm'] for v in cal['inputs']])!=r['radius']:raise ValueError('Calibration not exactly training content/reference median')
    for folder,record in [(candidate,r),(control,c),(cal_path.parent,cal)]:
        for name,sha in record['source_sha256'].items():
            if digest(folder/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Source snapshot changed')
    out=base/'radius-context-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in sorted(set(r['source_sha256'])|{'audit_radius_context.py','audit_consistent_reference.py'}):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(source_sha256={p.name:digest(p) for p in frozen.iterdir()},candidate_results_sha256=digest(candidate/'results.json'),control_results_sha256=digest(control/'results.json'),calibration_results_sha256=digest(cal_path),calibration_exposures_verified=2048,calibration_roles=['content','reference'],calibrated_radius=r['radius'],prepared_files_verified=len(prep['files']),cells={},native={},finished=False,app_promoted=False)
    def read(relative):
        with Image.open(data/lookup[relative]['prepared_path']) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    truth={v['index']:read(v['gt']) for v in rows};content={v['index']:read(v['content']) for v in rows};reference={v['index']:read(v['reference']) for v in rows};ids={v['index'] for v in rows};rng=np.random.default_rng(r['config']['seed']);orders=[hashlib.sha256(json.dumps([train[i]['index'] for batch in grouped_batches(train,rng) for i in batch]).encode()).hexdigest() for epoch in range(8)];initial={}
    for name,path,record in [('candidate',candidate,r),('control',control,c)]:
        initial[name]=torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True);means=[]
        for label in [('reused-control' if name=='candidate' else 'baseline-current'),*[f'epoch-{i}' for i in range(9)]]:
            scores=load(path/label/'scores.json');predictions={}
            if len(scores['cases'])!=64 or {v['index'] for v in scores['cases']}!=ids:raise ValueError('Incomplete score table')
            for item in scores['cases']:
                p=path/label/f'{item["index"]}.png'
                if digest(p)!=item['sha256']:raise ValueError('PNG changed')
                with Image.open(p) as im:predictions[item['index']]=np.array(im.convert('RGB'))
            actual=score_outputs(rows,predictions,truth)
            for v in actual['cases']:v['sha256']=digest(path/label/f'{v["index"]}.png')
            if actual!=scores:raise ValueError('Full RGB/paired/tone table differs')
            if label.startswith('epoch-'):means.append(scores['mean'])
        if int(np.argmin(means))!=record['best_epoch'] or means[record['best_epoch']]!=record['best_validation_mean'] or digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Selection/weights differ')
        for i,epoch in enumerate(record['epochs']):
            p=path/f'epoch-{i+1}.pth'
            if epoch['epoch']!=i+1 or epoch['updates']!=(i+1)*256 or epoch['validation_mean']!=means[i+1] or epoch['target_order_sha256']!=orders[i] or digest(p)!=epoch['checkpoint_sha256'] or any(not torch.isfinite(v).all() for v in torch.load(p,map_location='cpu',weights_only=True).values()):raise ValueError('Budget/order/checkpoint differs')
        report['cells'][name]=dict(png_verified=640,full_score_tables_recomputed=10,best_epoch=record['best_epoch'],mean=means[record['best_epoch']])
    if digest(candidate/'epoch-0.pth')!=digest(control/'epoch-0.pth') or set(initial['candidate'])!=set(initial['control']) or any(not torch.equal(v,initial['control'][k]) for k,v in initial['candidate'].items()):raise ValueError('Complete initial states differ')
    for name in ['initial_raw_equivalence','reused_control_native']:
        if len(r[name])!=64 or {v['index'] for v in r[name]}!=ids or any(v['maximum_float_difference']!=0 for v in r[name]):raise ValueError('Native raw/grid bridge incomplete')
    model,api,_=author_model();initialize_replication_model(model,torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval().requires_grad_(False);tensor=lambda x:(torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda();pixels=lambda raw:np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8);sets={}
    modes=[('baseline',candidate,'epoch-0',initial['candidate'],True),('control',control,f'epoch-{c["best_epoch"]}',torch.load(control/f'epoch-{c["best_epoch"]}.pth',map_location='cpu',weights_only=True),False),('candidate',candidate,f'epoch-{r["best_epoch"]}',torch.load(candidate/f'epoch-{r["best_epoch"]}.pth',map_location='cpu',weights_only=True),True)]
    with torch.inference_mode():
        for name,path,label,state,normalized in modes:
            model.load_state_dict(state,strict=True);predictions={};raw_predictions={}
            for row in rows:
                i=row['index'];x,y=tensor(content[i]),tensor(reference[i]);array=pixels(radius_forward(model,api,x,y,r['radius'],normalized)[0])
                with Image.open(path/label/f'{i}.png') as im:saved=np.array(im.convert('RGB'))
                if not np.array_equal(array,saved):raise ValueError('Native selected RGB differs')
                predictions[i]=array
                if name=='candidate':raw_predictions[i]=pixels(radius_forward(model,api,x,y,r['radius'],normalized=False)[0]);Image.fromarray(raw_predictions[i]).save(out/f'{i}-raw-mode.png')
            scores=score_outputs(rows,predictions,truth)
            for v in scores['cases']:v['sha256']=digest(path/label/f'{v["index"]}.png')
            if scores!=load(path/label/'scores.json'):raise ValueError('Native selected full metrics differ')
            sets[name]=(scores,predictions);report['native'][name]=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True)
            if name=='candidate':
                table=score_outputs(rows,raw_predictions,truth)
                for v in table['cases']:v['sha256']=digest(out/f'{v["index"]}-raw-mode.png')
                report['raw_mode_counterfactual']=dict(scores=table,mean_output_change=float(np.mean([np.abs(raw_predictions[i].astype(float)-predictions[i]).mean() for i in predictions])))
            print(json.dumps(dict(native=name,verified=True)),flush=True)
    gate=response_development_gate(sets['baseline'][0],sets['candidate'][0],sets['control'][0])
    if gate!=r['development_gate'] or sets['baseline'][0]!=r['baseline'] or r['baseline']!=c['baseline']:raise ValueError('Gate/baseline differs')
    for row in rows:
        i=row['index'];canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
        for j,(label,array) in enumerate([('Target',content[i]),('Reference',reference[i]),('GT',truth[i]),('Current13',sets['baseline'][1][i]),('Raw C10 control',sets['control'][1][i]),('Fixed-radius candidate',sets['candidate'][1][i])]):
            x=j%3*256;y=j//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(array),(x,y+19))
        canvas.save(out/f'{i}.jpg',quality=95)
    if digest(candidate/'results.json')!=report['candidate_results_sha256'] or digest(control/'results.json')!=report['control_results_sha256'] or digest(cal_path)!=report['calibration_results_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Evidence changed during audit')
    report.update(finished=True,passed=True,total_epoch_png_verified=1280,all_initial_states_identical=True,target_orders_recomputed=True,development_gate=gate);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','raw_mode_counterfactual']}),flush=True)


if __name__=='__main__':main()
