"""Audit the fixed four-cell study; real-reference metrics decide qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context
from appearance_pairwise import enable_appearance_context,appearance_forward
from appearance_invariance import invariance_gate
from consistent_reference import use_synthetic_reference
from pairwise_gate import require_completed_control,response_development_gate
from paired_training import grouped_batches
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';generated_dir=base/'cohort-consistent-reference-v1'
    paths={'C00':base/'appearance-pairwise-replica-control-v1','C10':base/'appearance-pairwise-replica-v1','C01':base/'appearance-invariance-C01-v2','C11':base/'appearance-invariance-C11-v2'}
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));records={name:load(p/'results.json') for name,p in paths.items()};config=records['C00']['config']
    for name,record in records.items():
        require_completed_control(record,config,record['source_sha256'])
        if record['appearance_enabled']!=(name[1]=='1') or record.get('augmentation_enabled',False)!=(name[2]=='1'):raise ValueError('Wrong factor flags')
    if config['seed']!=20261124 or config['appearance_init_seed']!=20261125 or records['C00']['source_sha256']!=records['C10']['source_sha256'] or records['C01']['source_sha256']!=records['C11']['source_sha256']:raise ValueError('Changed paired recipe')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];train=cohort['groups']['train'];ids={r['index'] for r in rows};lookup={r['original_path']:r for r in prep['files']}
    if len(ids)!=64 or not prep['passed'] or any(r['cohort_sha256']!=digest(cohort_path) or r['prepared_manifest_sha256']!=digest(prep_path) or r['baseline']!=records['C00']['baseline'] for r in records.values()):raise ValueError('Data or baseline differs')
    reused={name:dict(results_sha256=digest(paths[name]/'results.json'),selected_scores_sha256=digest(paths[name]/f'epoch-{records[name]["best_epoch"]}/scores.json'),selected_checkpoint_sha256=digest(paths[name]/f'epoch-{records[name]["best_epoch"]}.pth'),initial_checkpoint_sha256=digest(paths[name]/'epoch-0.pth')) for name in ['C00','C10']}
    for name in ['C01','C11']:
        record=records[name]
        if record['cell']!=name or record['reused_cells']!=reused or not record['initial_complete_state_exact_to_C00'] or not record['restored_initial_after_reuse']:raise ValueError('Invalid completed-cell reuse')
        for parent in ['C00','C10']:
            checks=record['reused_native'][parent]
            if len(checks)!=64 or {r['index'] for r in checks}!=ids or any(r['maximum_code_difference']!=0 for r in checks):raise ValueError('Invalid reuse-native record')
    if records['C11']['c01_snapshot']!=dict(results_sha256=digest(paths['C01']/'results.json'),selected_checkpoint_sha256=records['C01']['selected_checkpoint_sha256']):raise ValueError('C01 changed')
    generated=load(generated_dir/'results.json');gen={r['index']:r for r in generated['cases']};train_ids={r['index']:r for r in train};train_contents={r['content'] for r in train};parents={}
    if not generated['finished'] or len(generated['cases'])!=1024 or set(gen)!=set(train_ids) or generated['cohort_sha256']!=digest(cohort_path) or generated['prepared_manifest_sha256']!=digest(prep_path) or any(records[n]['generation_manifest_sha256']!=digest(generated_dir/'results.json') for n in ['C01','C11']):raise ValueError('Changed generated inputs')
    for item in generated['cases']:
        row=train_ids[item['index']];expected={row['content'],row['gt'],item['reference_original']}
        if item['content']!=row['content'] or item['gt']!=row['gt'] or item['reference_original'] not in train_contents or item['reference_original']==row['content'] or set(item['parent_sha256'])!=expected or item['generated_reference']!=f'{row["index"]}.png':raise ValueError('Generated provenance invalid')
        for p,sha in item['parent_sha256'].items():
            if lookup[p]['sha256']!=sha:raise ValueError('Parent manifest changed')
            parents[p]=sha
        if digest(generated_dir/item['generated_reference'])!=item['sha256']:raise ValueError('Generated PNG changed')
    for p,sha in parents.items():
        if digest(data/lookup[p]['prepared_path'])!=sha:raise ValueError('Parent bytes changed')
    for filename,sha in generated['source_sha256'].items():
        if digest(generated_dir/'frozen-source'/filename)!=sha:raise ValueError('Generation frozen source changed')
    out=base/'appearance-invariance-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=sorted(set(['audit_appearance_invariance.py','audit_consistent_reference.py','train_consistent_reference.py','capacity_audit.py',*[n for r in records.values() for n in r['source_sha256']]]))
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Fixed2x2 consumed-development study, not independent or product acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},cells={},generated_png_verified=1024,unique_parent_hashes_verified=len(parents),finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    orders=[];assignments=[];rng=np.random.default_rng(config['seed'])
    for epoch in range(1,9):
        ordered=[train[i] for batch in grouped_batches(train,rng) for i in batch];orders.append(hashlib.sha256(json.dumps([r['index'] for r in ordered]).encode()).hexdigest())
        values=[[r['index'],use_synthetic_reference(r['style_index'],epoch),gen[r['index']]['sha256'] if use_synthetic_reference(r['style_index'],epoch) else lookup[r['reference']]['sha256']] for r in ordered];assignments.append(hashlib.sha256(json.dumps(values).encode()).hexdigest())
    initial={}
    for name,path in paths.items():
        record=records[name]
        for filename,sha in record['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Frozen/executing source changed')
        rawchecks=record['initial_raw_equivalence']
        if len(rawchecks)!=64 or {r['index'] for r in rawchecks}!=ids or any(r['maximum_float_difference']!=0 for r in rawchecks):raise ValueError('Invalid original raw-check record')
        means=[];count=0
        for label in ['baseline-current',*[f'epoch-{i}' for i in range(9)]]:
            table=load(path/label/'scores.json')
            if len(table['cases'])!=64 or {r['index'] for r in table['cases']}!=ids or float(np.mean([r['mae'] for r in table['cases']]))!=table['mean']:raise ValueError('Incomplete saved scores')
            for row in table['cases']:
                if digest(path/label/f'{row["index"]}.png')!=row['sha256']:raise ValueError('Epoch PNG changed')
                count+=1
            if label!='baseline-current':means.append(table['mean'])
        if int(np.argmin(means))!=record['best_epoch'] or digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Invalid selection/checkpoint')
        initial[name]=torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True)
        for i,epoch in enumerate(record['epochs']):
            if digest(path/f'epoch-{epoch["epoch"]}.pth')!=epoch['checkpoint_sha256'] or epoch['validation_mean']!=means[epoch['epoch']] or epoch['target_order_sha256']!=orders[i]:raise ValueError('Epoch order/weight/score changed')
            if name in ['C01','C11'] and (epoch['reference_assignment_sha256']!=assignments[i] or epoch['synthetic_reference_count']!=512):raise ValueError('Actual reference assignments differ')
            if name[1]=='0':
                state=torch.load(path/f'epoch-{epoch["epoch"]}.pth',map_location='cpu',weights_only=True)
                if not epoch['unused_appearance_unchanged'] or any(not torch.equal(state[k],initial[name][k]) for k in initial[name] if k.startswith('reference_adapter.')):raise ValueError('Unused appearance parameters changed')
        report['cells'][name]=dict(png_verified=count,best_epoch=record['best_epoch'],mean=means[record['best_epoch']])
    for state in initial.values():
        if set(state)!=set(initial['C00']) or any(not torch.equal(v,initial['C00'][k]) for k,v in state.items()):raise ValueError('Four complete initial states differ')
    base_path=base/'multiref-control-v1/epoch-6.pth';base_state=torch.load(base_path,map_location='cpu',weights_only=True)
    if digest(base_path)!=generated['checkpoint_sha256'] or any(digest(base_path)!=r['start_checkpoint_sha256'] for r in records.values()) or any(not torch.equal(v,initial['C00'][k]) for k,v in base_state.items()):raise ValueError('Initial base differs from current13')
    def read(relative):
        entry=lookup[relative];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Prepared input changed')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    content={r['index']:read(r['content']) for r in rows};reference={r['index']:read(r['reference']) for r in rows};truth={r['index']:read(r['gt']) for r in rows}
    model,api,_=author_model();enable_pairwise_context(model);enable_appearance_context(model);model.eval().requires_grad_(False)
    tensor=lambda rgb:(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda();pixels=lambda x:np.rint(x[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    sets={};report['native_reproduction']={};report['appearance_layout_tests']=[]
    modes=[('baseline',paths['C00'],'baseline-current',initial['C00'],True),*[(name,path,f'epoch-{records[name]["best_epoch"]}',torch.load(path/f'epoch-{records[name]["best_epoch"]}.pth',map_location='cpu',weights_only=True),name[1]=='0') for name,path in paths.items()]]
    with torch.inference_mode():
        for name,path,label,state,disabled in modes:
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected state')
            model.load_state_dict(state,strict=True);predictions={};ablations={'appearance_off':{},'cyclic_appearance_reference':{}}
            for i,row in enumerate(rows):
                index=row['index'];c,s=tensor(content[index]),tensor(reference[index]);actual=pixels(appearance_forward(model,api,c,s,disable_appearance=disabled)[0])
                with Image.open(path/label/f'{index}.png') as im:saved=np.array(im.convert('RGB'))
                if int(np.abs(actual.astype(int)-saved.astype(int)).max())!=0:raise ValueError('Native selected RGB differs')
                predictions[index]=actual
                if name=='C11':
                    other=rows[(i+1)%len(rows)]['index'];off=pixels(appearance_forward(model,api,c,s,disable_appearance=True)[0]);cyclic=pixels(appearance_forward(model,api,c,s,appearance_reference=tensor(reference[other]))[0])
                    for mode,array in [('appearance_off',off),('cyclic_appearance_reference',cyclic)]:ablations[mode][index]=array;Image.fromarray(array).save(out/f'{index}-{mode}.png')
                    if i<4:
                        order=np.random.default_rng(20261123+i).permutation(256*256);image=tensor(reference[index].reshape(-1,3)[order].reshape(256,256,3));changed=pixels(appearance_forward(model,api,c,s,appearance_reference=image)[0]);off_changed=pixels(appearance_forward(model,api,c,image,disable_appearance=True)[0]);delta=int(np.abs(off.astype(int)-off_changed.astype(int)).max())
                        if delta!=0:raise ValueError('Histogram-only shuffle differs')
                        report['appearance_layout_tests'].append(dict(index=index,full_output_mean_code_change=float(np.abs(actual.astype(float)-changed).mean()),appearance_off_max_code_change=delta))
            table=score_outputs(rows,predictions,truth)
            for row in table['cases']:row['sha256']=digest(path/label/f'{row["index"]}.png')
            if table!=load(path/label/'scores.json'):raise ValueError('All selected metrics do not reproduce from images/GT')
            sets[name]=(table,predictions);report['native_reproduction'][name]=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True,mean=table['mean'])
            if name=='C11':
                report['ablations']={}
                for mode,arrays in ablations.items():
                    table=score_outputs(rows,arrays,truth)
                    for r in table['cases']:r['sha256']=digest(out/f'{r["index"]}-{mode}.png')
                    table['mean_output_change_from_full']=float(np.mean([np.abs(arrays[r['index']].astype(float)-predictions[r['index']]).mean() for r in rows]));report['ablations'][mode]=table
    baseline=sets['baseline'][0]
    if any(baseline!=r['baseline'] for r in records.values()):raise ValueError('Frozen current baseline differs')
    gates={'C00':response_development_gate(baseline,sets['C00'][0]),'C10':response_development_gate(baseline,sets['C10'][0],sets['C00'][0])}
    for name in ['C01','C11']:
        gate=invariance_gate(baseline,sets[name][0],sets['C00'][0],sets['C10'][0]);eligible=name=='C11';gates[name]=dict(gate,measured_gate_passed=gate['proceed_to_independent_evaluation'],candidate_cell=eligible,proceed_to_independent_evaluation=bool(eligible and gate['proceed_to_independent_evaluation']))
    if any(gates[n]!=records[n]['development_gate'] for n in records):raise ValueError('Complete gates differ')
    mean=lambda n:sets[n][0]['mean'];delta=lambda n:sets[n][0]['delta_mean']
    report['factor_effects']=dict(rgb_augmentation_without_appearance=mean('C01')-mean('C00'),rgb_augmentation_with_appearance=mean('C11')-mean('C10'),rgb_interaction=(mean('C11')-mean('C10'))-(mean('C01')-mean('C00')),delta_augmentation_without_appearance=delta('C01')-delta('C00'),delta_augmentation_with_appearance=delta('C11')-delta('C10'))
    for row in rows:
        index=row['index'];canvas=Image.new('RGB',(1024,550),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',content[index]),('Real reference',reference[index]),('GT',truth[index]),('Current13',sets['baseline'][1][index]),*[(label,sets[n][1][index]) for n,label in [('C00','C00: hist real'),('C10','C10: appearance real'),('C01','C01: hist augmented'),('C11','C11: appearance augmented')]]]
        for i,(label,im) in enumerate(panels):
            x=i%4*256;y=i//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(im),(x,y+19))
        canvas.save(out/f'{index}.jpg',quality=95)
    report.update(finished=True,passed=True,total_epoch_png_verified=2560,ablation_png_saved=128,all_four_initial_states_identical=True,target_orders_recomputed=True,reference_assignments_recomputed=True,gates=gates)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','ablations']}),flush=True)

if __name__=='__main__':main()
