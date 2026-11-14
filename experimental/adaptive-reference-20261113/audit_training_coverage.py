"""Recompute all equal-update trial scores/orders and selected native outputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from coverage_training import block_batches,combined_gate
from coverage_asset_plan import COHORT_SHA
from audit_consistent_reference import score_outputs
from pairwise_gate import require_completed_control,response_development_gate


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v1';prior=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    paths={k:base/f'training-coverage-{k}-v1' for k in ('control','expanded')};records={k:load(p/'results.json') for k,p in paths.items()};c,r=records['control'],records['expanded']
    cohort=load(data/'cohort.json');old=load(base/'cohort-multiref-v1/cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']};rows=cohort['groups']['validation'];ids={v['index'] for v in rows}
    if digest(data/'cohort.json')!=COHORT_SHA or not prep['passed'] or any(prep[k] for k in ('invalid_pairs','cross_split_exact_duplicates','cross_split_near_duplicates')) or len(ids)!=64 or rows!=old['groups']['validation']:raise ValueError('Changed/unaccepted data')
    if c['source_sha256']!=r['source_sha256'] or c['expanded'] or not r['expanded'] or c['passes_per_block']!=4 or r['passes_per_block']!=1:raise ValueError('Wrong study arms')
    for key in ('cohort_sha256','prepared_manifest_sha256','start_checkpoint_sha256','original_results_sha256','initial_checkpoint_sha256'):
        if c[key]!=r[key]:raise ValueError('Different shared initialization/data')
    if c['cohort_sha256']!=COHORT_SHA or c['prepared_manifest_sha256']!=digest(data/'srgb-v1/manifest.json') or c['start_checkpoint_sha256']!=digest(prior/'epoch-7.pth') or c['original_results_sha256']!=digest(prior/'results.json'):raise ValueError('Changed shared evidence')
    snap=dict(results_sha256=digest(paths['control']/'results.json'),selected_checkpoint_sha256=c['selected_checkpoint_sha256'],scores_sha256=digest(paths['control']/f'epoch-{c["best_epoch"]}/scores.json'))
    if r['control_snapshot']!=snap:raise ValueError('Matched control snapshot differs')
    for item in prep['files']:
        if digest(data/item['prepared_path'])!=item['sha256']:raise ValueError('Prepared input changed')
    for label,record in records.items():
        require_completed_control(record,c['config'],c['source_sha256'])
        for name,sha in record['source_sha256'].items():
            if digest(paths[label]/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Source changed')
        for name,sha in record['protected_inputs_sha256'].items():
            if digest(base/name)!=sha:raise ValueError('Protected evidence changed')
        if not record['initial_complete_state_exact'] or len(record['initial_native_raw_grid'])!=64 or {v['index'] for v in record['initial_native_raw_grid']}!=ids or any(v['maximum_float_difference']!=0 for v in record['initial_native_raw_grid']):raise ValueError('Initial bridge incomplete')
    out=base/'training-coverage-audit-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in sorted(set(r['source_sha256'])|{'audit_training_coverage.py'}):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(source_sha256={p.name:digest(p) for p in frozen.iterdir()},results_sha256={k:digest(p/'results.json') for k,p in paths.items()},prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),cells={},native={},finished=False,app_promoted=False)
    def image(relative):
        with Image.open(data/lookup[relative]['prepared_path']) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    truth={v['index']:image(v['gt']) for v in rows};content={v['index']:image(v['content']) for v in rows};reference={v['index']:image(v['reference']) for v in rows}
    def scores_from_png(directory):
        saved=load(directory/'scores.json');predictions={}
        if len(saved['cases'])!=64 or {v['index'] for v in saved['cases']}!=ids:raise ValueError('Incomplete score table')
        for item in saved['cases']:
            p=directory/f'{item["index"]}.png'
            if digest(p)!=item['sha256']:raise ValueError('Score PNG changed')
            with Image.open(p) as im:predictions[item['index']]=np.array(im.convert('RGB'))
        actual=score_outputs(rows,predictions,truth)
        for v in actual['cases']:v['sha256']=digest(directory/f'{v["index"]}.png')
        if actual!=saved:raise ValueError('RGB/paired/tone scores differ')
        return actual,predictions
    current,_=scores_from_png(prior/'baseline-current');original,_=scores_from_png(prior/'epoch-7');initial=torch.load(prior/'epoch-7.pth',map_location='cpu',weights_only=True);selected={}
    for label,record in records.items():
        directory=paths[label];train=old['groups']['train'] if label=='control' else cohort['groups']['train'];rng=np.random.default_rng(record['config']['seed']);means=[]
        for epoch in range(9):
            path=directory/f'epoch-{epoch}.pth';state=torch.load(path,map_location='cpu',weights_only=True)
            if set(state)!=set(initial) or any(v.shape!=initial[k].shape or v.dtype!=initial[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('Invalid checkpoint')
            scores,_=scores_from_png(directory/f'epoch-{epoch}');means.append(scores['mean'])
            if epoch==0:
                if digest(path)!=record['initial_checkpoint_sha256'] or any(not torch.equal(v,initial[k]) for k,v in state.items()) or scores!=original or scores!=record['warm_start_scores']:raise ValueError('Complete common initial state/scores differ')
            else:
                order=[train[i]['index'] for batch in block_batches(train,rng,record['passes_per_block']) for i in batch];entry=record['epochs'][epoch-1]
                if len(order)!=4096 or set(order)!={v['index'] for v in train} or set(Counter(order).values())!={record['passes_per_block']} or hashlib.sha256(json.dumps(order).encode()).hexdigest()!=entry['target_order_sha256'] or entry['updates']!=epoch*1024 or entry['checkpoint_sha256']!=digest(path) or entry['validation_mean']!=scores['mean'] or entry['delta_mean']!=scores['delta_mean']:raise ValueError('Budget/exposure/order/checkpoint differs')
        best=int(np.argmin(means))
        if best!=record['best_epoch'] or means[best]!=record['best_validation_mean'] or digest(directory/f'epoch-{best}.pth')!=record['selected_checkpoint_sha256'] or record['baseline']!=current:raise ValueError('Selection/baseline differs')
        selected[label]=scores_from_png(directory/f'epoch-{best}')[0];report['cells'][label]=dict(epoch_png_verified=576,full_score_tables_recomputed=9,best_epoch=best,mean=means[best],updates=8192)
    model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));current_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.eval().requires_grad_(False);tensor=lambda x:(torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda();native={}
    modes=[('current13',prior/'baseline-current',current_state),('originalC10',prior/'epoch-7',initial),*[(k,paths[k]/f'epoch-{records[k]["best_epoch"]}',torch.load(paths[k]/f'epoch-{records[k]["best_epoch"]}.pth',map_location='cpu',weights_only=True)) for k in ('control','expanded')]]
    with torch.inference_mode():
        for label,directory,state in modes:
            model.load_state_dict(state,strict=True);predictions={}
            for row in rows:
                i=row['index'];raw=appearance_forward(model,api,tensor(content[i]),tensor(reference[i]))[0];array=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                with Image.open(directory/f'{i}.png') as im:saved=np.array(im.convert('RGB'))
                if not np.array_equal(array,saved):raise ValueError('Native selected output differs')
                predictions[i]=array
            actual=score_outputs(rows,predictions,truth)
            for case in actual['cases']:case['sha256']=digest(directory/f'{case["index"]}.png')
            if actual!=load(directory/'scores.json'):raise ValueError('Native full metrics differ')
            native[label]=(actual,predictions);report['native'][label]=dict(cases=64,maximum_code_difference=0,full_scores_exact=True);print('native verified',label,flush=True)
    for label,record in records.items():
        original_gate=response_development_gate(current,selected[label],original);matched=response_development_gate(current,selected[label],selected['control']) if label=='expanded' else original_gate;final=combined_gate(matched,original_gate,label=='expanded')
        if final!=record['development_gate'] or original_gate!=record['original_C10_gate'] or matched!=record['matched_control_gate'] or final['proceed_to_independent_evaluation']!=record['proceed_to_independent_evaluation']:raise ValueError('Combined gates differ')
    for row in rows:
        i=row['index'];canvas=Image.new('RGB',(1024,550),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',content[i]),('Reference',reference[i]),('GT',truth[i]),('Current13',native['current13'][1][i]),('Original C10',native['originalC10'][1][i]),('Old data continuation',native['control'][1][i]),('Expanded data',native['expanded'][1][i])]
        for j,(label,array) in enumerate(panels):
            x=j%4*256;y=j//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(array),(x,y+19))
        canvas.save(out/f'{i}.jpg',quality=95)
    if any(digest(paths[k]/'results.json')!=sha for k,sha in report['results_sha256'].items()) or digest(data/'srgb-v1/manifest.json')!=report['prepared_manifest_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Audit evidence changed')
    report.update(finished=True,passed=True,epoch_png_verified=1152,original_baseline_png_verified=128,full_score_tables_recomputed=20,native_outputs_verified=256,development_gate=r['development_gate']);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
