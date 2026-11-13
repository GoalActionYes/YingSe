"""Audit completed matched runs and report secondary trained-statistic scores."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from pairwise_gate import require_completed_control,response_development_gate
from tone_relations import tone_relation_losses


def main():
    torch.set_num_threads(4);base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1'
    runs={name:base/f'tone-{suffix}-v1' for name,suffix in [('control','control'),('relations','relations')]}
    records={name:json.loads((directory/'results.json').read_text(encoding='utf-8')) for name,directory in runs.items()}
    a,b=records['control'],records['relations'];config=a['config']
    for record in records.values():require_completed_control(record,config,a['source_sha256'])
    if a['relations'] or not b['relations'] or a['baseline']!=b['baseline'] or a['cohort_sha256']!=b['cohort_sha256'] or a['prepared_manifest_sha256']!=b['prepared_manifest_sha256'] or not b['initial_state_exact_to_control']:raise ValueError('Arms differ outside the declared loss')
    if digest(runs['control']/'results.json')!=b['control_snapshot']['results_sha256'] or digest(runs['control']/f'epoch-{a["best_epoch"]}/scores.json')!=b['control_snapshot']['scores_sha256']:raise ValueError('Control snapshot changed')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or digest(cohort_path)!=a['cohort_sha256'] or digest(prep_path)!=a['prepared_manifest_sha256']:raise ValueError('Changed data')
    out=base/'tone-study-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['audit_tone_study.py','tone_relations.py','pairwise_gate.py','tone-relations-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed development outputs; secondary trained-descriptor scores are not independent quality evidence',source_sha256={p.name:digest(p) for p in frozen.iterdir()},arms={},finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');selected={};expected_ids={r['index'] for r in cohort['groups']['validation']}
    for name,directory in runs.items():
        record=records[name]
        for filename,sha in record['source_sha256'].items():
            if digest(directory/'frozen-source'/filename)!=sha:raise ValueError('Changed frozen source')
        scores_by_epoch=[];checked=0
        for epoch in range(9):
            scores=json.loads((directory/f'epoch-{epoch}/scores.json').read_text(encoding='utf-8'))
            if len(scores['cases'])!=64 or {r['index'] for r in scores['cases']}!=expected_ids:raise ValueError('Incomplete epoch scores')
            if float(np.mean([r['mae'] for r in scores['cases']]))!=scores['mean']:raise ValueError('Incorrect epoch mean')
            for row in scores['cases']:
                if digest(directory/f'epoch-{epoch}'/f'{row["index"]}.png')!=row['sha256']:raise ValueError('Changed epoch output')
                checked+=1
            if epoch and (record['epochs'][epoch-1]['validation_mean']!=scores['mean'] or digest(directory/f'epoch-{epoch}.pth')!=record['epochs'][epoch-1]['checkpoint_sha256']):raise ValueError('Changed epoch record or checkpoint')
            scores_by_epoch.append(scores)
        if record['best_epoch']!=int(np.argmin([s['mean'] for s in scores_by_epoch])) or digest(directory/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Selection differs from minimum MAE')
        if digest(directory/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or len(record['initial_equivalence'])!=64 or any(r['maximum_float_difference']!=0 for r in record['initial_equivalence']):raise ValueError('Changed initial equivalence')
        selected[name]=scores_by_epoch[record['best_epoch']]
        report['arms'][name]=dict(png_verified=checked,epochs=8,updates=2048,selected_epoch=record['best_epoch'],selected_mean=selected[name]['mean'],results_sha256=digest(directory/'results.json'))
    states=[torch.load(directory/'epoch-0.pth',map_location='cpu',weights_only=True) for directory in runs.values()]
    if set(states[0])!=set(states[1]) or any(not torch.equal(value,states[1][key]) for key,value in states[0].items()):raise ValueError('Initial parameter tensors differ')
    if response_development_gate(a['baseline'],selected['control'])!=a['development_gate'] or response_development_gate(b['baseline'],selected['relations'],selected['control'])!=b['development_gate']:raise ValueError('Stored gate does not recompute')
    lookup={r['original_path']:r for r in prep['files']};current_dir=base/'multiref-control-v1/epoch-6';current=json.loads((current_dir/'scores.json').read_text(encoding='utf-8'))
    if digest(current_dir/'scores.json')!=a['anchor_scores_sha256']:raise ValueError('Current model scores changed')
    paths={'current':current_dir,**{name:directory/f'epoch-{records[name]["best_epoch"]}' for name,directory in runs.items()}}
    expected={name:{r['index']:r for r in scores['cases']} for name,scores in [('current',current),*selected.items()]};groups={};all_mae={name:[] for name in paths}
    for row in cohort['groups']['validation']:
        inputs=[]
        for role in ['content','gt']:
            entry=lookup[row[role]];path=data/entry['prepared_path']
            if digest(path)!=entry['sha256']:raise ValueError('Changed prepared target or GT')
            with Image.open(path) as image:rgb=np.array(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            inputs.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
        images={}
        for name,directory in paths.items():
            path=directory/f'{row["index"]}.png'
            if digest(path)!=expected[name][row['index']]['sha256']:raise ValueError('Changed selected PNG')
            with Image.open(path) as image:rgb=np.array(image.convert('RGB'))
            gt=np.rint(inputs[1].permute(1,2,0).numpy()*255).astype(np.uint8);mae=float(np.abs(rgb.astype(float)-gt).mean())
            if mae!=expected[name][row['index']]['mae']:raise ValueError('Selected MAE does not reproduce against GT')
            all_mae[name].append(mae);images[name]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        groups.setdefault(row['group_id'],{})[row['style_index']]=(inputs,images)
    if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete descriptor groups')
    names=list(groups);losses={name:[] for name in paths}
    with torch.inference_mode():
        for offset in range(0,32,2):
            rows=[groups[names[i]][style] for i in [offset,offset+1] for style in [0,1]];content=torch.stack([r[0][0] for r in rows]);gt=torch.stack([r[0][1] for r in rows])
            for name in paths:losses[name].append([float(v) for v in tone_relation_losses(torch.stack([r[1][name] for r in rows]),gt,content)])
    report.update(finished=True,passed=True,total_png_verified=1152,initial_full_parameters_identical=True,stored_gates_reproduced=True,
        recomputed_selected_mae={name:float(np.mean(values)) for name,values in all_mae.items()},descriptor_scores={name:dict(absolute=float(np.mean(values,axis=0)[0]),paired=float(np.mean(values,axis=0)[1]),combined=float(np.mean(values,axis=0)@np.array([.25,.5]))) for name,values in losses.items()},
        descriptor_scope='Quantized RGB8 selected outputs, 32 development pairs; only secondary diagnostics, not model selection')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
