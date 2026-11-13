"""Selected-model ablations with true marginal reference histograms retained."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context,pairwise_forward,pairwise_histograms
from multiref_response import response_metrics
from tone_response import tone_response
from training_checks import finite_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='pairwise-context-v1');parser.add_argument('--control',default='pairwise-control-v1');parser.add_argument('--run',default='diagnostics-v1');args=parser.parse_args()
    for name in [args.candidate,args.control,args.run]:
        if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('Expected single directory names')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';run=base/args.candidate;control_dir=base/args.control;data=base/'cohort-multiref-v1'
    record=json.loads((run/'results.json').read_text(encoding='utf-8'));control=json.loads((control_dir/'results.json').read_text(encoding='utf-8'))
    for result in [record,control]:
        if not result['finished'] or 'development_gate' not in result or 'selected_checkpoint_sha256' not in result or result['epochs'][-1]['updates']!=2048:raise ValueError('Incomplete training evidence')
    if record['source_sha256']!=control['source_sha256']:raise ValueError('Arms do not have identical shared source snapshots')
    if digest(control_dir/'results.json')!=record['control_snapshot']['results_sha256']:raise ValueError('Control changed')
    control_scores_path=control_dir/f'epoch-{control["best_epoch"]}/scores.json';anchor_path=base/'multiref-control-v1/epoch-6/scores.json'
    if digest(control_scores_path)!=record['control_snapshot']['scores_sha256'] or digest(anchor_path)!=record['anchor_scores_sha256']:raise ValueError('Comparison score evidence changed')
    control_png={r['index']:r['sha256'] for r in json.loads(control_scores_path.read_text(encoding='utf-8'))['cases']}
    anchor_png={r['index']:r['sha256'] for r in json.loads(anchor_path.read_text(encoding='utf-8'))['cases']}
    if digest(control_dir/f'epoch-{control["best_epoch"]}.pth')!=control['selected_checkpoint_sha256']:raise ValueError('Control checkpoint changed')
    epoch=record['best_epoch'];checkpoint=run/f'epoch-{epoch}.pth'
    if digest(checkpoint)!=record['selected_checkpoint_sha256']:raise ValueError('Changed candidate checkpoint')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or digest(cohort_path)!=record['cohort_sha256'] or digest(prep_path)!=record['prepared_manifest_sha256']:raise ValueError('Changed development data')
    scores=json.loads((run/f'epoch-{epoch}/scores.json').read_text(encoding='utf-8'));expected={r['index']:r['sha256'] for r in scores['cases']}
    protocol='tone-relations-protocol.md' if 'relations' in record else 'pairwise-training-protocol.md'
    if digest(Path(__file__).with_name(protocol))!=record['source_sha256'][protocol]:raise ValueError('Training protocol changed')
    out=run/args.run;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_pairwise_context.py','pairwise_context.py',protocol,'multiref_response.py','tone_response.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed development only; retained true marginal histograms, cyclic mismatch of joint reference features only; no semantic or independent acceptance',checkpoint_sha256=digest(checkpoint),
        matched_frozen_source=True,start_checkpoint_sha256=record['start_checkpoint_sha256'],control_checkpoint_sha256=control['selected_checkpoint_sha256'],
        anchor_scores_sha256=digest(anchor_path),control_scores_sha256=digest(control_scores_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],layout_controls=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');lookup={r['original_path']:r for r in prep['files']}
    def read(name):
        e=lookup[name];path=data/e['prepared_path']
        if digest(path)!=e['sha256']:raise ValueError('Changed prepared image')
        with Image.open(path) as image:return np.array(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    samples=[(r,*[read(r[k]) for k in ['content','reference','gt']]) for r in cohort['groups']['validation']]
    model,api,_=author_model();enable_pairwise_context(model);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    def tensor(rgb):return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(1,3,256,256))[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    modes=['full','no_pairwise','zero_reference','wrong_reference'];groups={mode:{} for mode in modes}
    with torch.inference_mode():
        references=[pairwise_histograms(api['srgb_to_ycbcr'](tensor(s))) for _,_,s,_ in samples]
        for i,(row,c,s,gt) in enumerate(samples):
            ct,st=tensor(c),tensor(s);images={'full':pixels(pairwise_forward(model,api,ct,st)[0]),'no_pairwise':pixels(pairwise_forward(model,api,ct,st,True)[0])}
            for mode,replacement in [('zero_reference',torch.zeros_like(references[i])),('wrong_reference',references[(i+1)%len(samples)])]:
                handle=model.pairwise_adapter.register_forward_pre_hook(lambda module,args,replacement=replacement:(args[0],replacement))
                try:images[mode]=pixels(pairwise_forward(model,api,ct,st)[0])
                finally:handle.remove()
            saved_path=run/f'epoch-{epoch}/{row["index"]}.png'
            if digest(saved_path)!=expected[row['index']]:raise ValueError('Changed selected saved prediction')
            with Image.open(saved_path) as im:saved=np.array(im.convert('RGB'))
            delta=int(np.abs(images['full'].astype(int)-saved.astype(int)).max())
            if delta!=0:raise ValueError('Selected prediction not exactly reproduced')
            case=dict(index=row['index'],scores={k:float(np.abs(v.astype(float)-gt).mean()) for k,v in images.items()},reproduction_max_rgb8_delta=delta,
                wrong_joint_reference_response=float(np.abs(images['full'].astype(float)-images['wrong_reference']).mean()),
                wrong_joint_donor=dict(index=samples[(i+1)%len(samples)][0]['index'],group_id=samples[(i+1)%len(samples)][0]['group_id'],style_index=samples[(i+1)%len(samples)][0]['style_index']))
            report['cases'].append(case)
            for mode,image in images.items():groups[mode].setdefault(row['group_id'],{})[row['style_index']]=(image,gt)
            current=base/'multiref-control-v1/epoch-6'/f'{row["index"]}.png';matched=control_dir/f'epoch-{control["best_epoch"]}'/f'{row["index"]}.png'
            if digest(current)!=anchor_png[row['index']] or digest(matched)!=control_png[row['index']]:raise ValueError('Changed comparison PNG')
            with Image.open(current) as im:a=np.array(im.convert('RGB'))
            with Image.open(matched) as im:b=np.array(im.convert('RGB'))
            canvas=Image.new('RGB',(768,825),'#222');draw=ImageDraw.Draw(canvas)
            candidate_label='Tone relation training' if record.get('relations') else 'Pairwise context'
            for j,(label,image) in enumerate([('Input',c),('Reference',s),('GT',gt),('Current 13',a),('Matched more steps',b),(candidate_label,images['full']),('No pairwise head',images['no_pairwise']),('Zero joint reference',images['zero_reference']),('Wrong joint reference',images['wrong_reference'])]):
                x=j%3*256;y=j//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(image),(x,y+19))
            canvas.save(out/f'{row["index"]}.jpg',quality=95)
            if i<4:
                rng=np.random.default_rng(20261118+i);shuffled=s.reshape(-1,3)[rng.permutation(256*256)].reshape(s.shape);other=tensor(shuffled)
                same_joint=torch.equal(references[i],pairwise_histograms(api['srgb_to_ycbcr'](other)));same_marginal=torch.equal(api['batch_histogram_1D'](api['srgb_to_ycbcr'](st)),api['batch_histogram_1D'](api['srgb_to_ycbcr'](other)))
                permuted=pixels(pairwise_forward(model,api,ct,other)[0]);report['layout_controls'].append(dict(index=row['index'],same_joint=same_joint,same_marginal=same_marginal,maximum_output_difference=int(np.abs(permuted.astype(int)-images['full'].astype(int)).max())))
    report['summaries']={}
    for mode in modes:
        if len(groups[mode])!=32 or any(set(g)!={0,1} for g in groups[mode].values()):raise ValueError('Incomplete ablation groups')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups[mode].items()]
        report['summaries'][mode]=dict(mean=float(np.mean([r['scores'][mode] for r in report['cases']])),delta_mean=float(np.mean([p['delta_mae'] for p in paired])),
            relationship_error=float(np.mean([(p['tones']['absolute_delta_error']['light_minus_dark_cb']+p['tones']['absolute_delta_error']['light_minus_dark_cr'])/2 for p in paired])),paired=paired)
    report.update(finished=True,exact_reproduction=True,mean_wrong_joint_reference_response=float(np.mean([r['wrong_joint_reference_response'] for r in report['cases']])))
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summaries={k:{n:v for n,v in s.items() if n!='paired'} for k,s in report['summaries'].items()},mean_wrong_joint_reference_response=report['mean_wrong_joint_reference_response'],layout_controls=report['layout_controls'],exact_reproduction=True)),flush=True)


if __name__=='__main__':main()
