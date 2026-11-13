"""Audit the fixed auxiliary-loss candidate against its stronger C10 control."""
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
from paired_training import grouped_batches
from pairwise_gate import response_development_gate
from audit_consistent_reference import score_outputs


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';candidate=base/'paired-style-contrast-v1';control=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    records={name:load(p/'results.json') for name,p in [('candidate',candidate),('control',control)]};r,c=records['candidate'],records['control'];config=r['config']
    if not r['finished'] or r['app_promoted'] or not c['finished'] or config!={**c['config'],'style_weight':.01,'temperature':.1}:raise ValueError('Wrong/incomplete study')
    snapshot=dict(results_sha256=digest(control/'results.json'),scores_sha256=digest(control/f'epoch-{c["best_epoch"]}/scores.json'),initial_sha256=digest(control/'epoch-0.pth'),selected_sha256=digest(control/f'epoch-{c["best_epoch"]}.pth'))
    if r['control_snapshot']!=snapshot or digest(base/'real-style-correspondence-v1/results.json')!=r['diagnostic_results_sha256']:raise ValueError('Changed control/diagnostic snapshot')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);lookup={v['original_path']:v for v in prep['files']};train=cohort['groups']['train'];rows=cohort['groups']['validation']
    if any(record['cohort_sha256']!=digest(cohort_path) or record['prepared_manifest_sha256']!=digest(prep_path) or record['start_checkpoint_sha256']!=digest(base/'multiref-control-v1/epoch-6.pth') for record in records.values()):raise ValueError('Changed cohort/base')
    for entry in prep['files']:
        if digest(data/entry['prepared_path'])!=entry['sha256']:raise ValueError('Changed prepared input')
    for name,path in [('candidate',candidate),('control',control)]:
        for filename,sha in records[name]['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Changed executable study source')
    out=base/'paired-style-contrast-audit-v1';out.mkdir(exist_ok=False)
    audit_sources=['audit_paired_style_contrast.py','audit_consistent_reference.py']
    for filename in audit_sources:(out/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    report=dict(candidate_results_sha256=digest(candidate/'results.json'),control_results_sha256=digest(control/'results.json'),source_sha256={n:digest(out/n) for n in audit_sources},prepared_files_verified=len(prep['files']),cells={},native={},finished=False,app_promoted=False)
    def read(relative):
        with Image.open(data/lookup[relative]['prepared_path']) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    truth={v['index']:read(v['gt']) for v in rows};content={v['index']:read(v['content']) for v in rows};reference={v['index']:read(v['reference']) for v in rows}
    ids={v['index'] for v in rows};rng=np.random.default_rng(config['seed']);orders=[]
    for epoch in range(8):
        order=[train[i]['index'] for batch in grouped_batches(train,rng) for i in batch];orders.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
    initial={}
    for name,path in [('candidate',candidate),('control',control)]:
        record=records[name];means=[];count=0;initial[name]=torch.load(path/'epoch-0.pth',map_location='cpu',weights_only=True)
        if len(record['epochs'])!=8 or digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256']:raise ValueError('Incomplete initial/epoch records')
        for label in [('reused-control' if name=='candidate' else 'baseline-current'),*[f'epoch-{i}' for i in range(9)]]:
            scores=load(path/label/'scores.json')
            if len(scores['cases'])!=64 or {v['index'] for v in scores['cases']}!=ids:raise ValueError('Incomplete output table')
            for item in scores['cases']:
                p=path/label/f'{item["index"]}.png'
                if digest(p)!=item['sha256']:raise ValueError('PNG changed')
                with Image.open(p) as im:pixels=np.array(im.convert('RGB'))
                if pixels.shape!=(256,256,3) or float(np.abs(pixels.astype(float)-truth[item['index']]).mean())!=item['mae']:raise ValueError('Per-image RGB MAE differs')
                count+=1
            if float(np.mean([v['mae'] for v in scores['cases']]))!=scores['mean']:raise ValueError('Mean differs')
            if label.startswith('epoch-'):means.append(scores['mean'])
        if int(np.argmin(means))!=record['best_epoch'] or digest(path/f'epoch-{record["best_epoch"]}.pth')!=record['selected_checkpoint_sha256']:raise ValueError('Selection/checkpoint differs')
        for i,epoch in enumerate(record['epochs']):
            if epoch['epoch']!=i+1 or epoch['updates']!=(i+1)*256 or epoch['validation_mean']!=means[i+1] or epoch['target_order_sha256']!=orders[i] or digest(path/f'epoch-{i+1}.pth')!=epoch['checkpoint_sha256']:raise ValueError('Training budget/order/weights differ')
            if name=='candidate' and not np.isclose(epoch['train_loss'],epoch['rgb_l1']+.1*epoch['gradient_l1']+.01*epoch['style_ce'],rtol=1e-6,atol=1e-8):raise ValueError('Recorded loss components inconsistent')
        report['cells'][name]=dict(png_verified=count,best_epoch=record['best_epoch'],mean=means[record['best_epoch']])
    if digest(candidate/'epoch-0.pth')!=digest(control/'epoch-0.pth') or set(initial['candidate'])!=set(initial['control']) or any(not torch.equal(v,initial['control'][k]) for k,v in initial['candidate'].items()):raise ValueError('Complete initial states differ')
    if len(r['initial_raw_equivalence'])!=64 or any(v['maximum_float_difference']!=0 for v in r['initial_raw_equivalence']) or {v['index'] for v in r['initial_raw_equivalence']}!=ids or not np.isfinite(r['initial_auxiliary_encoder_gradient_norm']) or r['initial_auxiliary_encoder_gradient_norm']<=0:raise ValueError('Initial raw/gradient checks incomplete')
    model,api,_=author_model();enable_pairwise_context(model);enable_appearance_context(model);model.eval().requires_grad_(False);sets={};tensor=lambda v:(torch.from_numpy(v.copy()).permute(2,0,1)[None].float()/255).cuda()
    modes=[('baseline',candidate,'epoch-0',initial['candidate']),('control',control,f'epoch-{c["best_epoch"]}',torch.load(control/f'epoch-{c["best_epoch"]}.pth',map_location='cpu',weights_only=True)),('candidate',candidate,f'epoch-{r["best_epoch"]}',torch.load(candidate/f'epoch-{r["best_epoch"]}.pth',map_location='cpu',weights_only=True))]
    with torch.inference_mode():
        for name,path,label,state in modes:
            if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected state')
            model.load_state_dict(state,strict=True);predictions={}
            for row in rows:
                i=row['index'];raw=appearance_forward(model,api,tensor(content[i]),tensor(reference[i]))[0];pixels=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                with Image.open(path/label/f'{i}.png') as im:saved=np.array(im.convert('RGB'))
                if not np.array_equal(pixels,saved):raise ValueError('Selected native output differs')
                predictions[i]=pixels
            scores=score_outputs(rows,predictions,truth)
            for row in scores['cases']:row['sha256']=digest(path/label/f'{row["index"]}.png')
            if scores!=load(path/label/'scores.json'):raise ValueError('Selected full metrics differ')
            sets[name]=(scores,predictions);report['native'][name]=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True);print(json.dumps(dict(native=name,verified=True)),flush=True)
    if sets['baseline'][0]!=r['baseline'] or r['baseline']!=c['baseline']:raise ValueError('Current baseline differs')
    gate=response_development_gate(sets['baseline'][0],sets['candidate'][0],sets['control'][0])
    if gate!=r['development_gate']:raise ValueError('Gate differs')
    for row in rows:
        i=row['index'];canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
        for j,(label,array) in enumerate([('Target',content[i]),('Reference',reference[i]),('GT',truth[i]),('Current13',sets['baseline'][1][i]),('C10 control',sets['control'][1][i]),('Paired style candidate',sets['candidate'][1][i])]):
            x=j%3*256;y=j//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(array),(x,y+19))
        canvas.save(out/f'{i}.jpg',quality=95)
    if digest(candidate/'results.json')!=report['candidate_results_sha256'] or digest(control/'results.json')!=report['control_results_sha256']:raise ValueError('Records changed during audit')
    report.update(development_gate=gate,finished=True,passed=True,all_initial_state_exact=True,target_orders_recomputed=True,total_epoch_png_verified=1280)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
