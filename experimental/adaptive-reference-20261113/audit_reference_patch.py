"""Full frozen-state/PNG/native audit for the reference feature-token study."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from reference_patch import enable_reference_patch,patch_forward,patch_development_gate
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from paired_training import grouped_batches
from pairwise_gate import require_completed_control,response_development_gate
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';paths={name:base/f'reference-patch-{name}-v1' for name in ['mean','tokens']};load=lambda p:json.loads(p.read_text(encoding='utf-8'));records={n:load(p/'results.json') for n,p in paths.items()};c,r=records['mean'],records['tokens'];config=r['config']
    for record in records.values():require_completed_control(record,config,r['source_sha256'])
    if c['use_tokens'] or not r['use_tokens'] or c['baseline']!=r['baseline'] or c['external_C10_snapshot']!=r['external_C10_snapshot']:raise ValueError('Wrong/mismatched arms')
    snapshot=dict(results_sha256=digest(paths['mean']/'results.json'),scores_sha256=digest(paths['mean']/f'epoch-{c["best_epoch"]}/scores.json'),checkpoint_sha256=digest(paths['mean']/f'epoch-{c["best_epoch"]}.pth'),initial_sha256=digest(paths['mean']/'epoch-0.pth'))
    if snapshot!=r['mean_control_snapshot']:raise ValueError('Changed matched mean control')
    external_dir=base/'appearance-pairwise-replica-v1';external=load(external_dir/'results.json');external_weight=external_dir/f'epoch-{external["best_epoch"]}.pth';external_scores=external_dir/f'epoch-{external["best_epoch"]}/scores.json';es=r['external_C10_snapshot']
    if dict(results_sha256=digest(external_dir/'results.json'),scores_sha256=digest(external_scores),checkpoint_sha256=digest(external_weight))!=es:raise ValueError('Changed C10 benchmark')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];train=cohort['groups']['train'];lookup={v['original_path']:v for v in prep['files']};base_path=base/'multiref-control-v1/epoch-6.pth'
    if any(record['cohort_sha256']!=digest(cohort_path) or record['prepared_manifest_sha256']!=digest(prep_path) or record['start_checkpoint_sha256']!=digest(base_path) for record in records.values()):raise ValueError('Study inputs differ')
    for item in prep['files']:
        if digest(data/item['prepared_path'])!=item['sha256']:raise ValueError('Prepared asset changed')
    for name,path in paths.items():
        for filename,sha in records[name]['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Frozen/executing source differs')
    out=base/'reference-patch-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in sorted(set(r['source_sha256'])|{'audit_reference_patch.py','audit_consistent_reference.py'}):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(source_sha256={p.name:digest(p) for p in frozen.iterdir()},results_sha256={n:digest(p/'results.json') for n,p in paths.items()},prepared_files_verified=len(prep['files']),cells={},native={},ablations={},finished=False,app_promoted=False)
    def read(relative):
        with Image.open(data/lookup[relative]['prepared_path']) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    truth={v['index']:read(v['gt']) for v in rows};content={v['index']:read(v['content']) for v in rows};reference={v['index']:read(v['reference']) for v in rows};ids={v['index'] for v in rows};initial={};rng=np.random.default_rng(config['seed']);orders=[]
    for epoch in range(8):orders.append(hashlib.sha256(json.dumps([train[i]['index'] for batch in grouped_batches(train,rng) for i in batch]).encode()).hexdigest())
    def score_files(path,label):
        saved=load(path/label/'scores.json');arrays={}
        if len(saved['cases'])!=64 or {v['index'] for v in saved['cases']}!=ids:raise ValueError('Incomplete output cases')
        for v in saved['cases']:
            p=path/label/f'{v["index"]}.png'
            if digest(p)!=v['sha256']:raise ValueError('PNG changed')
            with Image.open(p) as im:arrays[v['index']]=np.array(im.convert('RGB'))
        actual=score_outputs(rows,arrays,truth)
        for v in actual['cases']:v['sha256']=digest(path/label/f'{v["index"]}.png')
        if actual!=saved:raise ValueError('Full RGB/paired/tone table does not reproduce')
        return actual
    for name,path in paths.items():
        record=records[name];means=[]
        for label in ['baseline-current',*[f'epoch-{i}' for i in range(9)]]:
            table=score_files(path,label)
            if label!='baseline-current':means.append(table['mean'])
        if int(np.argmin(means))!=record['best_epoch'] or means[record['best_epoch']]!=record['best_validation_mean']:raise ValueError('Selection differs')
        initial[name]=torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True)
        if digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Initial/selected weights changed')
        for i,epoch in enumerate(record['epochs']):
            p=path/f'epoch-{i+1}.pth'
            if epoch['epoch']!=i+1 or epoch['updates']!=256*(i+1) or epoch['validation_mean']!=means[i+1] or epoch['target_order_sha256']!=orders[i] or digest(p)!=epoch['checkpoint_sha256']:raise ValueError('Budget/order/weight differs')
            if any(not torch.isfinite(v).all() for v in torch.load(p,map_location='cpu',weights_only=True).values()):raise ValueError('Nonfinite epoch state')
        if record['new_parameters']!=65664 or len(record['initial_raw_equivalence'])!=64 or {v['index'] for v in record['initial_raw_equivalence']}!=ids or any(v['maximum_float_difference']!=0 for v in record['initial_raw_equivalence']):raise ValueError('Incomplete initial bridge')
        report['cells'][name]=dict(png_verified=640,full_score_tables_recomputed=10,best_epoch=record['best_epoch'],mean=means[record['best_epoch']],maximum_branch_gradient={k:max(e['branch_gradient_maximum'][k] for e in record['epochs']) for k in ['q','key','value','projection']})
    if digest(paths['mean']/'epoch-0.pth')!=digest(paths['tokens']/'epoch-0.pth') or set(initial['mean'])!=set(initial['tokens']) or any(not torch.equal(v,initial['mean'][k]) for k,v in initial['tokens'].items()):raise ValueError('Complete arm initial state differs')
    base_state=torch.load(base_path,map_location='cpu',weights_only=True);extra={f'reference_patch.{name}' for name in ['q.weight','kv.weight','proj.weight','proj.bias']}
    if set(initial['mean'])-set(base_state)!=extra or any(not torch.equal(v,initial['mean'][k]) for k,v in base_state.items()):raise ValueError('Shared initial model is not current13')
    model,api,_=author_model();enable_reference_patch(model);model.eval().requires_grad_(False);sets={};tensor=lambda x:(torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda();pixels=lambda raw:np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    modes=[('baseline',paths['mean'],'baseline-current',initial['mean'],True),*[(n,p,f'epoch-{records[n]["best_epoch"]}',torch.load(p/f'epoch-{records[n]["best_epoch"]}.pth',map_location='cpu',weights_only=True),n=='mean') for n,p in paths.items()]]
    with torch.inference_mode():
        for name,path,label,state,pooled in modes:
            model.load_state_dict(state,strict=True);predictions={};ablations={mode:{} for mode in ['collapsed_tokens','disabled_branch']}
            for row in rows:
                i=row['index'];c0,s0=tensor(content[i]),tensor(reference[i]);raw,_=patch_forward(model,api,c0,s0,pooled=pooled);actual=pixels(raw)
                with Image.open(path/label/f'{i}.png') as im:saved=np.array(im.convert('RGB'))
                if not np.array_equal(actual,saved):raise ValueError('Selected native output differs')
                predictions[i]=actual
                if name=='tokens':
                    for mode,array in [('collapsed_tokens',pixels(patch_forward(model,api,c0,s0,pooled=True)[0])),('disabled_branch',pixels(patch_forward(model,api,c0,s0,disabled=True)[0]))]:ablations[mode][i]=array;Image.fromarray(array).save(out/f'{i}-{mode}.png')
            scores=score_outputs(rows,predictions,truth)
            for v in scores['cases']:v['sha256']=digest(path/label/f'{v["index"]}.png')
            if scores!=load(path/label/'scores.json'):raise ValueError('Selected native full metrics differ')
            sets[name]=(scores,predictions);report['native'][name]=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True)
            if name=='tokens':
                for mode,arrays in ablations.items():
                    table=score_outputs(rows,arrays,truth)
                    for v in table['cases']:v['sha256']=digest(out/f'{v["index"]}-{mode}.png')
                    report['ablations'][mode]=dict(scores=table,mean_output_change=float(np.mean([np.abs(arrays[i].astype(float)-predictions[i]).mean() for i in predictions])))
            print(json.dumps(dict(native=name,verified=True)),flush=True)
    del model
    benchmark_model,benchmark_api,_=author_model();initialize_replication_model(benchmark_model,base_state);benchmark_model.load_state_dict(torch.load(external_weight,map_location='cpu',weights_only=True),strict=True);benchmark_model.eval().requires_grad_(False);predictions={}
    with torch.inference_mode():
        for row in rows:
            i=row['index'];array=pixels(appearance_forward(benchmark_model,benchmark_api,tensor(content[i]),tensor(reference[i]))[0]);p=external_dir/f'epoch-{external["best_epoch"]}'/f'{i}.png'
            with Image.open(p) as im:saved=np.array(im.convert('RGB'))
            if not np.array_equal(array,saved):raise ValueError('External C10 native differs')
            predictions[i]=array
    comparison=score_outputs(rows,predictions,truth)
    for v in comparison['cases']:v['sha256']=digest(external_dir/f'epoch-{external["best_epoch"]}'/f'{v["index"]}.png')
    if comparison!=load(external_scores) or sets['baseline'][0]!=c['baseline']:raise ValueError('External/baseline metrics differ')
    sets['C10']=(comparison,predictions);report['native']['C10']=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True)
    gates=dict(mean=dict(current_only=response_development_gate(sets['baseline'][0],sets['mean'][0]),external_C10=response_development_gate(sets['baseline'][0],sets['mean'][0],comparison),proceed_to_independent_evaluation=False),tokens=patch_development_gate(sets['baseline'][0],sets['tokens'][0],sets['mean'][0],comparison))
    if any(gates[n]!=records[n]['development_gate'] for n in records):raise ValueError('Full qualification differs')
    for row in rows:
        i=row['index'];canvas=Image.new('RGB',(1024,550),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',content[i]),('Reference',reference[i]),('GT',truth[i]),('Current13',sets['baseline'][1][i]),('External C10',sets['C10'][1][i]),('Mean-token control',sets['mean'][1][i]),('16-token candidate',sets['tokens'][1][i])]
        for j,(label,array) in enumerate(panels):x=j%4*256;y=j//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(array),(x,y+19))
        canvas.save(out/f'{i}.jpg',quality=95)
    if any(digest(p/'results.json')!=report['results_sha256'][n] for n,p in paths.items()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()) or digest(external_dir/'results.json')!=es['results_sha256'] or digest(external_scores)!=es['scores_sha256'] or digest(external_weight)!=es['checkpoint_sha256']:raise ValueError('Evidence changed during audit')
    report.update(finished=True,passed=True,total_epoch_png_verified=1280,ablation_png_saved=128,all_initial_states_identical=True,target_orders_recomputed=True,development_gates=gates);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','ablations']}),flush=True)


if __name__=='__main__':main()
