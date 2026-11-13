"""Audit reference supervision experiment, with paired metrics recomputed from PNG."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from pairwise_gate import require_completed_control,response_development_gate
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context
from train_consistent_reference import forward
from multiref_response import response_metrics
from tone_response import tone_response
from paired_training import grouped_batches
from capacity_audit import render_coefficients


def score_outputs(rows,predictions,truths):
    """Rebuild case and same-content response metrics from decoded pixel arrays."""
    cases=[];groups={}
    if len(rows)!=64 or len({r['index'] for r in rows})!=64:raise ValueError('Expected64 unique real-reference development cases')
    for row in rows:
        index=row['index'];prediction=predictions[index];truth=truths[index]
        cases.append(dict(index=index,mae=float(np.abs(prediction.astype(float)-truth).mean())))
        group=groups.setdefault(row['group_id'],{})
        if row['style_index'] in group:raise ValueError('Duplicate paired style')
        group[row['style_index']]=(prediction,truth)
    if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete response groups')
    paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
    return dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';runs={'control':base/'consistent-reference-control-v1','augmented':base/'consistent-reference-v1'};generated_dir=base/'cohort-consistent-reference-v1'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));records={k:load(p/'results.json') for k,p in runs.items()};control,candidate=records['control'],records['augmented'];generated=load(generated_dir/'results.json')
    for r in records.values():require_completed_control(r,control['config'],control['source_sha256'])
    if control['augment_reference'] or not candidate['augment_reference'] or control['baseline']!=candidate['baseline'] or not candidate['initial_state_exact_to_control']:raise ValueError('Mismatched study arms')
    for key in ['cohort_sha256','prepared_manifest_sha256','generated_manifest_sha256','start_checkpoint_sha256']:
        if control[key]!=candidate[key]:raise ValueError('Changed shared inputs')
    if digest(runs['control']/'results.json')!=candidate['control_snapshot']['results_sha256'] or digest(runs['control']/f'epoch-{control["best_epoch"]}/scores.json')!=candidate['control_snapshot']['scores_sha256']:raise ValueError('Changed matched control')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];train=cohort['groups']['train'];expected_ids={r['index'] for r in rows}
    if len(expected_ids)!=64 or not prep['passed'] or digest(cohort_path)!=control['cohort_sha256'] or digest(prep_path)!=control['prepared_manifest_sha256'] or digest(generated_dir/'results.json')!=control['generated_manifest_sha256'] or not generated['finished']:raise ValueError('Changed data')
    lookup={r['original_path']:r for r in prep['files']};train_ids={r['index']:r for r in train};train_contents={r['content'] for r in train};parent_hashes={};gen={r['index']:r for r in generated['cases']}
    if len(gen)!=1024 or len(generated['cases'])!=1024 or set(gen)!=set(train_ids):raise ValueError('Incomplete generated references')
    for item in generated['cases']:
        row=train_ids[item['index']];parents={row['content'],row['gt'],item['reference_original']}
        if item['content']!=row['content'] or item['gt']!=row['gt'] or item['reference_original'] not in train_contents or item['reference_original']==row['content'] or set(item['parent_sha256'])!=parents:raise ValueError('Generated provenance differs')
        for path,sha in item['parent_sha256'].items():
            if lookup[path]['sha256']!=sha:raise ValueError('Incorrect parent checksum')
            parent_hashes[path]=sha
        if digest(generated_dir/item['generated_reference'])!=item['sha256']:raise ValueError('Changed generated PNG')
    for path,sha in parent_hashes.items():
        if digest(data/lookup[path]['prepared_path'])!=sha:raise ValueError('Changed training parent')
    for name,sha in generated['source_sha256'].items():
        if digest(generated_dir/'frozen-source'/name)!=sha:raise ValueError('Changed generation frozen source')
    for name in ['capacity_audit.py','train_triplet_pilot.py']:
        if digest(Path(__file__).with_name(name))!=generated['source_sha256'][name]:raise ValueError('Current generation reproduction dependency changed')
    out=base/'consistent-reference-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['audit_consistent_reference.py','capacity_audit.py','independent_coco.py',*control['source_sha256']]:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed development audit; paired metrics recomputed from all selected PNGs and GT, no independent or product acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},arms={},generated_png_hashes_verified=1024,unique_training_parent_hashes_verified=len(parent_hashes),finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    order_rng=np.random.default_rng(control['config']['seed']);orders=[]
    for epoch in range(8):
        order=[train[i]['index'] for batch in grouped_batches(train,order_rng) for i in batch];orders.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
    for name,path in runs.items():
        record=records[name]
        for filename,sha in record['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Execution or frozen source differs from training')
        means=[];pngs=0
        for label in ['baseline-current',*[f'epoch-{i}' for i in range(9)]]:
            scores=load(path/label/'scores.json')
            if len(scores['cases'])!=64 or {r['index'] for r in scores['cases']}!=expected_ids or float(np.mean([r['mae'] for r in scores['cases']]))!=scores['mean']:raise ValueError('Invalid saved score table')
            for row in scores['cases']:
                if digest(path/label/f'{row["index"]}.png')!=row['sha256']:raise ValueError('Changed saved PNG')
                pngs+=1
            if label!='baseline-current':means.append(scores['mean'])
        if int(np.argmin(means))!=record['best_epoch'] or len(record['initial_full_equivalence'])!=64 or {r['index'] for r in record['initial_full_equivalence']}!=expected_ids or any(r['maximum_float_difference']!=0 for r in record['initial_full_equivalence']):raise ValueError('Invalid selection or initial record')
        if digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Changed selected/initial weights')
        for i,row in enumerate(record['epochs']):
            if digest(path/f'epoch-{row["epoch"]}.pth')!=row['checkpoint_sha256'] or row['validation_mean']!=means[row['epoch']] or row['target_order_sha256']!=orders[i] or row['synthetic_references']!=(0 if name=='control' else 512):raise ValueError('Changed epoch, order or replacement budget')
        report['arms'][name]=dict(png_verified=pngs,selected_epoch=record['best_epoch'],selected_mean=means[record['best_epoch']])
    states=[torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True) for path in runs.values()]
    if set(states[0])!=set(states[1]) or any(not torch.equal(v,states[1][k]) for k,v in states[0].items()):raise ValueError('Initial full states differ')
    checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(checkpoint)!=control['start_checkpoint_sha256'] or digest(checkpoint)!=generated['checkpoint_sha256']:raise ValueError('Changed base guide weights')
    initial_base=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.equal(v,states[0][k]) for k,v in initial_base.items()):raise ValueError('Initial base parameters differ from current13')
    def read(relative):
        item=lookup[relative];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('Changed prepared input')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    truths={r['index']:read(r['gt']) for r in rows};content={r['index']:read(r['content']) for r in rows};references={r['index']:read(r['reference']) for r in rows}
    model,api,_=author_model();model.load_state_dict(initial_base,strict=True);model.eval().requires_grad_(False)
    selected_generation=[generated['cases'][0]['index'],generated['cases'][1]['index'],max(generated['cases'],key=lambda r:r['target_approximation_mae'])['index'],max(generated['cases'],key=lambda r:r['generated_endpoint_fraction'])['index']]
    report['generated_native_spot_checks']=[]
    with torch.inference_mode():
        for index in dict.fromkeys(selected_generation):
            item=gen[index];actual=render_coefficients(model,api,read(item['reference_original']),np.array(item['coefficients']))
            with Image.open(generated_dir/item['generated_reference']) as im:saved=np.array(im.convert('RGB'))
            delta=int(np.abs(actual.astype(int)-saved.astype(int)).max())
            if delta!=0:raise ValueError('Selected generation does not reproduce')
            report['generated_native_spot_checks'].append(dict(index=index,maximum_code_difference=delta))
    enable_pairwise_context(model);model.eval().requires_grad_(False)
    tensor=lambda rgb:(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    pixels=lambda raw:np.rint(raw[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    sets={};report['native_reproduction']={};report['reference_layout_tests']=[]
    with torch.inference_mode():
        for name,path,label,state in [('baseline',runs['control'],'baseline-current',states[0]),*[(name,path,f'epoch-{records[name]["best_epoch"]}',torch.load(path/f'epoch-{records[name]["best_epoch"]}.pth',map_location='cpu',weights_only=True)) for name,path in runs.items()]]:
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected state')
            model.load_state_dict(state,strict=True);predictions={};stored=load(path/label/'scores.json')
            for number,row in enumerate(rows):
                index=row['index'];c,s=tensor(content[index]),tensor(references[index]);actual=pixels(forward(model,api,c,s)[0])
                with Image.open(path/label/f'{index}.png') as im:saved=np.array(im.convert('RGB'))
                delta=int(np.abs(actual.astype(int)-saved.astype(int)).max())
                if delta!=0:raise ValueError('Native output does not reproduce')
                predictions[index]=actual
                if name=='augmented' and number<4:
                    order=np.random.default_rng(20261122+number).permutation(256*256);shuffled=references[index].reshape(-1,3)[order].reshape(256,256,3);other=pixels(forward(model,api,c,tensor(shuffled))[0]);difference=int(np.abs(actual.astype(int)-other.astype(int)).max())
                    if difference!=0:raise ValueError('Unexpected reference permutation difference')
                    report['reference_layout_tests'].append(dict(index=index,maximum_code_difference=difference))
            rebuilt=score_outputs(rows,predictions,truths)
            for row in rebuilt['cases']:row['sha256']=digest(path/label/f'{row["index"]}.png')
            if rebuilt!=stored:raise ValueError('Selected metrics do not exactly recompute from images')
            sets[name]=(rebuilt,predictions);report['native_reproduction'][name]=dict(cases=64,maximum_code_difference=0,all_saved_metrics_exactly_recomputed=True,mean=rebuilt['mean'])
    if sets['baseline'][0]!=control['baseline'] or sets['baseline'][0]!=candidate['baseline']:raise ValueError('Stored baseline differs')
    gates={name:response_development_gate(sets['baseline'][0],sets[name][0],sets['control'][0] if name=='augmented' else None) for name in runs}
    if any(gates[name]!=records[name]['development_gate'] for name in runs):raise ValueError('Fully recomputed gates differ')
    for row in rows:
        index=row['index'];canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
        for i,(label,im) in enumerate([('Target',content[index]),('Real reference',references[index]),('GT',truths[index]),('Current13',sets['baseline'][1][index]),('Matched control',sets['control'][1][index]),('Reference augmentation',sets['augmented'][1][index])]):
            x=i%3*256;y=i//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(im),(x,y+19))
        canvas.save(out/f'{index}.jpg',quality=95)
    report.update(finished=True,passed=True,total_png_verified=1280,all_initial_parameters_identical=True,target_orders_reproduced=True,all_selected_metrics_and_gates_recomputed=True,gates=gates)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
