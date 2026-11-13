"""Validate completed output-constraint study, native reproduction and maps."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from pairwise_gate import require_completed_control,response_development_gate
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context
from global_grid import projected_forward


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';runs={'control':base/'global-grid-control-v1','global':base/'global-grid-v1'}
    records={name:json.loads((path/'results.json').read_text(encoding='utf-8')) for name,path in runs.items()};a,b=records['control'],records['global']
    for record in records.values():require_completed_control(record,a['config'],a['source_sha256'])
    if a['global_output'] or not b['global_output'] or a['baseline']!=b['baseline'] or not b['initial_state_exact_to_control'] or a['cohort_sha256']!=b['cohort_sha256'] or a['prepared_manifest_sha256']!=b['prepared_manifest_sha256']:raise ValueError('Mismatched arms')
    if digest(runs['control']/'results.json')!=b['control_snapshot']['results_sha256'] or digest(runs['control']/f'epoch-{a["best_epoch"]}/scores.json')!=b['control_snapshot']['scores_sha256']:raise ValueError('Control changed')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or digest(cohort_path)!=a['cohort_sha256'] or digest(prep_path)!=a['prepared_manifest_sha256']:raise ValueError('Changed data')
    expected_ids={r['index'] for r in cohort['groups']['validation']};selected={};out=base/'global-grid-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['audit_global_grid.py','global_grid.py','pairwise_context.py','pairwise_gate.py','global-grid-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed development matched study; reproduction and fixed-map invariance, not independent or product acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},arms={},finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name,path in runs.items():
        record=records[name]
        for filename,sha in record['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha:raise ValueError('Frozen source changed')
        for filename in ['global_grid.py','pairwise_context.py','pairwise_gate.py']:
            if digest(Path(__file__).with_name(filename))!=record['source_sha256'][filename]:raise ValueError('Current execution code differs from trained source')
        epoch_means=[];checked=0
        for label in ['baseline-current',*[f'epoch-{i}' for i in range(9)]]:
            scores=json.loads((path/label/'scores.json').read_text(encoding='utf-8'))
            if len(scores['cases'])!=64 or {r['index'] for r in scores['cases']}!=expected_ids or float(np.mean([r['mae'] for r in scores['cases']]))!=scores['mean']:raise ValueError('Incomplete or incorrect scores')
            for row in scores['cases']:
                if digest(path/label/f'{row["index"]}.png')!=row['sha256']:raise ValueError('Changed PNG')
                checked+=1
            if label!='baseline-current':epoch_means.append(scores['mean'])
        if record['best_epoch']!=int(np.argmin(epoch_means)) or len(record['initial_full_equivalence'])!=64 or any(r['maximum_float_difference']!=0 for r in record['initial_full_equivalence']):raise ValueError('Invalid initial record or selection')
        for row in record['epochs']:
            if digest(path/f'epoch-{row["epoch"]}.pth')!=row['checkpoint_sha256'] or row['validation_mean']!=epoch_means[row['epoch']]:raise ValueError('Changed checkpoint or epoch mean')
        if digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Changed selected checkpoint')
        selected[name]=json.loads((path/f'epoch-{record["best_epoch"]}/scores.json').read_text(encoding='utf-8'));report['arms'][name]=dict(png_verified=checked,selected_epoch=record['best_epoch'],selected_mean=selected[name]['mean'],initial_output_mean=record['initial_output_mean'])
    states=[torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True) for path in runs.values()]
    if set(states[0])!=set(states[1]) or any(not torch.equal(value,states[1][key]) for key,value in states[0].items()):raise ValueError('Initial full parameters differ')
    if response_development_gate(a['baseline'],selected['control'])!=a['development_gate'] or response_development_gate(b['baseline'],selected['global'],selected['control'])!=b['development_gate']:raise ValueError('Gate does not reproduce')
    lookup={r['original_path']:r for r in prep['files']};samples=[]
    def read(relative):
        entry=lookup[relative];path=data/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed prepared input')
        with Image.open(path) as image:return np.array(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for row in cohort['groups']['validation']:samples.append((row,*[read(row[k]) for k in ['content','reference','gt']]))
    model,api,_=author_model();enable_pairwise_context(model);model.eval().requires_grad_(False)
    tensor=lambda rgb:(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    pixels=lambda raw:np.rint(raw[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    images={name:{} for name in runs};report['native_reproduction']={};report['fixed_map_tests']=[]
    with torch.inference_mode():
        for name,path in runs.items():
            record=records[name];state=torch.load(path/f'epoch-{record["best_epoch"]}.pth',map_location='cpu',weights_only=True)
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected weights')
            model.load_state_dict(state,strict=True);scores={r['index']:r for r in selected[name]['cases']};max_diff=0;maes=[]
            for i,(row,c,s,gt) in enumerate(samples):
                ct,st=tensor(c),tensor(s);raw,grid=projected_forward(model,api,ct,st,record['global_output']);actual=pixels(raw)
                expected_shape=(1,13,8,1,1) if name=='global' else (1,13,8,16,16)
                if tuple(grid.shape)!=expected_shape:raise ValueError('Unexpected selected grid shape')
                saved_path=path/f'epoch-{record["best_epoch"]}'/f'{row["index"]}.png'
                if digest(saved_path)!=scores[row['index']]['sha256']:raise ValueError('Changed selected PNG')
                with Image.open(saved_path) as image:saved=np.array(image.convert('RGB'))
                difference=int(np.abs(actual.astype(int)-saved.astype(int)).max());max_diff=max(max_diff,difference)
                if difference!=0:raise ValueError('Selected output not exactly reproduced')
                mae=float(np.abs(actual.astype(float)-gt).mean())
                if mae!=scores[row['index']]['mae']:raise ValueError('Selected MAE does not reproduce')
                maes.append(mae);images[name][row['index']]=actual
                if name=='global' and i<4:
                    order=np.random.default_rng(20261120+i).permutation(256*256);shuffled=c.reshape(-1,3)[order].reshape(c.shape);other=tensor(shuffled)
                    fixed=pixels(model.apply_coeffs(model.sampler(grid,other),other));expected=actual.reshape(-1,3)[order].reshape(c.shape);delta=int(np.abs(fixed.astype(int)-expected.astype(int)).max())
                    if delta!=0:raise ValueError('Fixed global map depends on position')
                    ref_shuffle=s.reshape(-1,3)[order].reshape(s.shape);permuted,other_grid=projected_forward(model,api,ct,tensor(ref_shuffle),True)
                    reference_delta=int(np.abs(pixels(permuted).astype(int)-actual.astype(int)).max())
                    if reference_delta!=0:raise ValueError('Reference permutation changes histogram-only conditioning')
                    report['fixed_map_tests'].append(dict(index=row['index'],position_permutation_max_code_difference=delta,reference_layout_max_code_difference=reference_delta))
            report['native_reproduction'][name]=dict(cases=64,maximum_code_difference=max_diff,recomputed_mean_mae=float(np.mean(maes)))
    current_dir=base/'multiref-control-v1/epoch-6';current_scores=json.loads((current_dir/'scores.json').read_text(encoding='utf-8'));current_sha={r['index']:r['sha256'] for r in current_scores['cases']}
    if digest(current_dir/'scores.json')!=a['anchor_scores_sha256']:raise ValueError('Current baseline changed')
    for row,c,s,gt in samples:
        path=current_dir/f'{row["index"]}.png'
        if digest(path)!=current_sha[row['index']]:raise ValueError('Changed current baseline PNG')
        with Image.open(path) as image:current=np.array(image.convert('RGB'))
        canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
        for i,(label,array) in enumerate([('Target',c),('Reference',s),('GT',gt),('Current 13',current),('Matched full grid',images['control'][row['index']]),('Predicted global grid',images['global'][row['index']])]):
            x=i%3*256;y=i//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(array),(x,y+19))
        canvas.save(out/f'{row["index"]}.jpg',quality=95)
    report.update(finished=True,passed=True,total_png_verified=1280,initial_full_parameters_identical=True,stored_gates_reproduced=True,app_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
