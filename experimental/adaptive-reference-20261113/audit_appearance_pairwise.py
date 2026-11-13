"""Audit reused-control appearance study and separate input-use from quality."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context
from paired_training import grouped_batches
from appearance_pairwise import enable_appearance_context,appearance_forward
from pairwise_gate import require_completed_control,response_development_gate
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';runs={'control':base/'consistent-reference-control-v1','candidate':base/'appearance-pairwise-v1'};bridge_dir=base/'appearance-pairwise-bridge-cpu-v3'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));records={name:load(path/'results.json') for name,path in runs.items()};control,candidate=records['control'],records['candidate'];bridge=load(bridge_dir/'results.json')
    for record in records.values():require_completed_control(record,control['config'],record['source_sha256'])
    snapshot=dict(results_sha256=digest(runs['control']/'results.json'),selected_scores_sha256=digest(runs['control']/f'epoch-{control["best_epoch"]}/scores.json'),initial_checkpoint_sha256=digest(runs['control']/'epoch-0.pth'),selected_checkpoint_sha256=digest(runs['control']/f'epoch-{control["best_epoch"]}.pth'))
    if control['augment_reference'] or not candidate['control_reused'] or candidate['control_snapshot']!=snapshot or bridge['control_snapshot']!=snapshot or not bridge['finished'] or not bridge['passed'] or candidate['bridge_results_sha256']!=digest(bridge_dir/'results.json'):raise ValueError('Changed reuse evidence')
    step=bridge['one_step_bridge']
    if step['device']!='cpu' or step['shared_parameter_maximum_difference']!=0 or step['control']!=step['disabled_appearance'] or not step['new_parameter_gradients_absent']:raise ValueError('Invalid exact CPU bridge')
    for name,sha in bridge['source_sha256'].items():
        if digest(bridge_dir/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed bridge source')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];expected_ids={r['index'] for r in rows}
    if len(expected_ids)!=64 or not prep['passed'] or any(record['cohort_sha256']!=digest(cohort_path) or record['prepared_manifest_sha256']!=digest(prep_path) for record in records.values()) or candidate['baseline']!=control['baseline']:raise ValueError('Changed common data/baseline')
    for label in ['initial_native','selected_control_native']:
        if len(bridge[label])!=64 or {r['index'] for r in bridge[label]}!=expected_ids or any(r['raw_grid_maximum_difference']!=0 or r['maximum_code_difference']!=0 for r in bridge[label]):raise ValueError('Invalid64-case bridge record')
    out=base/'appearance-pairwise-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=sorted(set(['audit_appearance_pairwise.py','audit_consistent_reference.py','independent_coco.py',*control['source_sha256'],*candidate['source_sha256']]))
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed real-reference development; reused control; selected outputs/metrics and ablations, not independent or product acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},arms={},finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rng=np.random.default_rng(control['config']['seed']);orders=[]
    for epoch in range(8):
        order=[cohort['groups']['train'][i]['index'] for batch in grouped_batches(cohort['groups']['train'],rng) for i in batch]
        orders.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
    for name,path in runs.items():
        record=records[name]
        for filename,sha in record['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Changed frozen/executing source')
        means=[];pngs=0
        for label in ['baseline-current',*[f'epoch-{i}' for i in range(9)]]:
            table=load(path/label/'scores.json')
            if len(table['cases'])!=64 or {r['index'] for r in table['cases']}!=expected_ids or table['mean']!=float(np.mean([r['mae'] for r in table['cases']])):raise ValueError('Invalid epoch score table')
            for row in table['cases']:
                if digest(path/label/f'{row["index"]}.png')!=row['sha256']:raise ValueError('Changed epoch PNG')
                pngs+=1
            if label!='baseline-current':means.append(table['mean'])
        if record['best_epoch']!=int(np.argmin(means)) or digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Invalid selection/checkpoints')
        for epoch in record['epochs']:
            if digest(path/f'epoch-{epoch["epoch"]}.pth')!=epoch['checkpoint_sha256'] or epoch['validation_mean']!=means[epoch['epoch']] or epoch['target_order_sha256']!=control['epochs'][epoch['epoch']-1]['target_order_sha256'] or epoch['target_order_sha256']!=orders[epoch['epoch']-1]:raise ValueError('Changed weight, order or score')
        report['arms'][name]=dict(png_verified=pngs,best_epoch=record['best_epoch'],mean=means[record['best_epoch']])
    old_initial=torch.load(runs['control']/'epoch-0.pth',map_location='cpu',weights_only=True);initial=torch.load(runs['candidate']/'epoch-0.pth',map_location='cpu',weights_only=True)
    if digest(runs['candidate']/'epoch-0.pth')!=bridge['candidate_initial_checkpoint_sha256'] or any(not torch.equal(v,initial[k]) for k,v in old_initial.items()):raise ValueError('Initial common state differs')
    added={f'reference_adapter.layers.{i}.{field}' for i in [0,2] for field in ['weight','bias']}
    if set(initial)-set(old_initial)!=added:raise ValueError('Unexpected initial model state')
    lookup={r['original_path']:r for r in prep['files']}
    def read(relative):
        item=lookup[relative];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('Changed prepared input')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    content={r['index']:read(r['content']) for r in rows};references={r['index']:read(r['reference']) for r in rows};truths={r['index']:read(r['gt']) for r in rows}
    model,api,_=author_model();enable_pairwise_context(model);enable_appearance_context(model);model.eval().requires_grad_(False)
    tensor=lambda rgb:(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    pixels=lambda x:np.rint(x[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    sets={};report['native_reproduction']={};report['appearance_layout_tests']=[]
    selected_control=torch.load(runs['control']/f'epoch-{control["best_epoch"]}.pth',map_location='cpu',weights_only=True);control_state={**initial,**selected_control};selected_candidate=torch.load(runs['candidate']/f'epoch-{candidate["best_epoch"]}.pth',map_location='cpu',weights_only=True)
    modes=[('baseline',runs['candidate'],'baseline-current',initial,False),('control',runs['control'],f'epoch-{control["best_epoch"]}',control_state,True),('candidate',runs['candidate'],f'epoch-{candidate["best_epoch"]}',selected_candidate,False)]
    with torch.inference_mode():
        for name,path,label,state,disabled in modes:
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected model')
            model.load_state_dict(state,strict=True);predictions={};ablations={'appearance_off':{},'wrong_appearance_reference':{}};stored=load(path/label/'scores.json')
            for i,row in enumerate(rows):
                index=row['index'];c,s=tensor(content[index]),tensor(references[index]);actual=pixels(appearance_forward(model,api,c,s,disable_appearance=disabled)[0])
                with Image.open(path/label/f'{index}.png') as im:saved=np.array(im.convert('RGB'))
                if int(np.abs(actual.astype(int)-saved.astype(int)).max())!=0:raise ValueError('Native selected output differs')
                predictions[index]=actual
                if name=='candidate':
                    other=rows[(i+1)%len(rows)]['index'];off=pixels(appearance_forward(model,api,c,s,disable_appearance=True)[0]);wrong=pixels(appearance_forward(model,api,c,s,appearance_reference=tensor(references[other]))[0])
                    for mode,array in [('appearance_off',off),('wrong_appearance_reference',wrong)]:
                        ablations[mode][index]=array;Image.fromarray(array).save(out/f'{index}-{mode}.png')
                    if i<4:
                        order=np.random.default_rng(20261123+i).permutation(256*256);shuffled=references[index].reshape(-1,3)[order].reshape(256,256,3);image=tensor(shuffled)
                        changed=pixels(appearance_forward(model,api,c,s,appearance_reference=image)[0]);off_shuffled=pixels(appearance_forward(model,api,c,image,disable_appearance=True)[0]);off_delta=int(np.abs(off.astype(int)-off_shuffled.astype(int)).max())
                        if off_delta!=0:raise ValueError('Reference shuffle changes histogram-only branch')
                        report['appearance_layout_tests'].append(dict(index=index,full_output_mean_code_change=float(np.abs(actual.astype(float)-changed).mean()),appearance_off_max_code_change=off_delta))
            rebuilt=score_outputs(rows,predictions,truths)
            for row in rebuilt['cases']:row['sha256']=digest(path/label/f'{row["index"]}.png')
            if rebuilt!=stored:raise ValueError('Selected RGB/paired/tone metrics differ from GT recomputation')
            sets[name]=(rebuilt,predictions);report['native_reproduction'][name]=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True,mean=rebuilt['mean'])
            if name=='candidate':
                report['ablations']={}
                for mode,arrays in ablations.items():
                    score=score_outputs(rows,arrays,truths)
                    for r in score['cases']:r['sha256']=digest(out/f'{r["index"]}-{mode}.png')
                    score['mean_output_change_from_full']=float(np.mean([np.abs(arrays[r['index']].astype(float)-predictions[r['index']]).mean() for r in rows]));report['ablations'][mode]=score
                report['wrong_appearance_assignment']=[dict(index=r['index'],appearance_index=rows[(i+1)%len(rows)]['index']) for i,r in enumerate(rows)]
    if sets['baseline'][0]!=control['baseline'] or sets['baseline'][0]!=candidate['baseline']:raise ValueError('Baseline mismatch')
    gates={'control':response_development_gate(sets['baseline'][0],sets['control'][0]),'candidate':response_development_gate(sets['baseline'][0],sets['candidate'][0],sets['control'][0])}
    if any(gates[name]!=records[name]['development_gate'] for name in gates):raise ValueError('Recomputed complete gates differ')
    for row in rows:
        index=row['index'];canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',content[index]),('Real reference',references[index]),('GT',truths[index]),('Current13',sets['baseline'][1][index]),('Reused matched control',sets['control'][1][index]),('Appearance + pairwise',sets['candidate'][1][index])]
        for i,(label,im) in enumerate(panels):
            x=i%3*256;y=i//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(im),(x,y+19))
        canvas.save(out/f'{index}.jpg',quality=95)
    report.update(finished=True,passed=True,total_epoch_png_verified=1280,ablation_png_saved=128,shared_initial_parameters_identical=True,target_orders_recomputed=True,gates=gates)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['ablations','wrong_appearance_assignment','source_sha256']}),flush=True)

if __name__=='__main__':main()
