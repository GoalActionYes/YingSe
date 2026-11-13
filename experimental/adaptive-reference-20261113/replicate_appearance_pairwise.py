"""One seed24 matched replication; both arms share all initial parameters."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from pairwise_gate import require_completed_control,response_development_gate
from paired_training import grouped_batches
from training_checks import finite_output
from multiref_response import response_metrics
from tone_response import tone_response

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--appearance',action='store_true');args=parser.parse_args()
    config=dict(seed=20261124,appearance_init_seed=20261125,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    def seed():
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'multiref-control-v1/epoch-6.pth';cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if digest(checkpoint)!=START_SHA or digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c':raise ValueError('Changed starting evidence')
    cohort,prep=load(cohort_path),load(prep_path)
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Invalid fixed cohort')
    lookup={r['original_path']:data/r['prepared_path'] for r in prep['files']}
    for row in prep['files']:
        if digest(lookup[row['original_path']])!=row['sha256']:raise ValueError('Changed prepared image')
    anchor_path=base/'multiref-control-v1/epoch-6/scores.json';anchor=load(anchor_path);anchor_rows={r['index']:r for r in anchor['cases']}
    out=base/('appearance-pairwise-replica-v1' if args.appearance else 'appearance-pairwise-replica-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['replicate_appearance_pairwise.py','appearance_replication.py','appearance_pairwise.py','appearance-pairwise-replication-protocol.md','reference_context.py','pairwise_context.py','paired_training.py','pairwise_gate.py','training_gates.py','training_checks.py','train_triplet_pilot.py','multiref_response.py','tone_response.py','independent_coco.py']
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='One predeclared seed24 replication on consumed real-reference development; no fresh pixels or product promotion',config=config,appearance_enabled=args.appearance,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),anchor_scores_sha256=digest(anchor_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},initial_raw_equivalence=[],epochs=[],finished=False,app_promoted=False)
    comparison=None;control_dir=base/'appearance-pairwise-replica-control-v1'
    if args.appearance:
        control_path=control_dir/'results.json';control=load(control_path);require_completed_control(control,config,report['source_sha256'])
        if control['appearance_enabled'] or any(control[k]!=report[k] for k in ['start_checkpoint_sha256','cohort_sha256','prepared_manifest_sha256','anchor_scores_sha256']):raise ValueError('Mismatched replication control')
        if digest(control_dir/f'epoch-{control["best_epoch"]}.pth')!=control['selected_checkpoint_sha256']:raise ValueError('Changed control checkpoint')
        scores_path=control_dir/f'epoch-{control["best_epoch"]}/scores.json';comparison=load(scores_path);report['control_snapshot']=dict(results_sha256=digest(control_path),selected_scores_sha256=digest(scores_path))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(r,*[image(lookup[r[role]]) for role in ['content','reference','gt']]) for r in cohort['groups'][part]] for part in ['train','validation']}
    inputs=lambda x:(x.float()/255).cuda()
    model,api,author=author_model();base_state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in base_state.values()):raise ValueError('Nonfinite current weights')
    model.load_state_dict(base_state,strict=True);model.eval();report['author_revision']=author['revision'];oracle=[]
    with torch.inference_mode():
        for row,c,s,gt in samples['validation']:
            c,s=inputs(c[None]),inputs(s[None]);hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));raw,grid=model(c,hc,hs)
            oracle.append((finite_output(raw,c.shape).cpu(),finite_output(grid,(1,13,8,16,16)).cpu()))
    initialize_replication_model(model,base_state)
    if args.appearance:
        initial=torch.load(control_dir/'epoch-0.pth',map_location='cpu',weights_only=True);state=model.state_dict()
        if set(initial)!=set(state) or any(not torch.equal(v.cpu(),initial[k]) for k,v in state.items()):raise ValueError('Complete initial state differs between arms')
        report['initial_complete_state_exact_to_control']=True
    initial_reference={k:v.detach().cpu().clone() for k,v in model.reference_adapter.state_dict().items()}
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(label):
        model.eval();directory=out/label;directory.mkdir();cases=[];groups={}
        for number,(row,c,s,gt) in enumerate(samples['validation']):
            raw,grid=appearance_forward(model,api,inputs(c[None]),inputs(s[None]),disable_appearance=not args.appearance);prediction=pixels(raw[0]);truth=gt.permute(1,2,0).numpy();p=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(p)
            if label=='baseline-current':
                difference=max(float((raw.cpu()-oracle[number][0]).abs().max()),float((grid.cpu()-oracle[number][1]).abs().max()))
                if difference!=0 or digest(p)!=anchor_rows[row['index']]['sha256']:raise ValueError('Initial raw/grid/PNG differs from current13')
                report['initial_raw_equivalence'].append(dict(index=row['index'],maximum_float_difference=difference))
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(p)));group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate validation style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete paired evaluation')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        result=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate('baseline-current');initial=evaluate('epoch-0');oracle.clear();report['baseline']=baseline
    if initial!=baseline or (args.appearance and baseline!=control['baseline']):raise ValueError('Initial/current score tables differ')
    torch.save(model.state_dict(),out/'epoch-0.pth');report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(baseline=baseline['mean'],appearance_enabled=args.appearance,training_seed=config['seed'])),flush=True)
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());identities={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    seed();rng=np.random.default_rng(config['seed']);train_rows=cohort['groups']['train'];updates=0;best=initial['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];order=[];start=time.perf_counter()
        for indices in grouped_batches(train_rows,rng):
            batch=[samples['train'][i] for i in indices];c,s,gt=[inputs(torch.stack([r[k] for r in batch])) for k in [1,2,3]];order.extend(r[0]['index'] for r in batch)
            optimizer.zero_grad(set_to_none=True);prediction=appearance_forward(model,api,c,s,disable_appearance=not args.appearance)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if not args.appearance and any(p.grad is not None for p in model.reference_adapter.parameters()):raise ValueError('Disabled appearance received gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:
                progress=dict(epoch=epoch,updates=updates,loss=float(loss));(out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if len(order)!=1024 or set(order)!={r['index'] for r in train_rows} or (args.appearance and order_sha!=control['epochs'][epoch-1]['target_order_sha256']):raise ValueError('Training stream differs')
        if not args.appearance and any(not torch.equal(v.detach().cpu(),initial_reference[k]) for k,v in model.reference_adapter.state_dict().items()):raise ValueError('Unused appearance parameters changed')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained state')
        p=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),p);validation=evaluate(f'epoch-{epoch}')
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        row=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(p),target_order_sha256=order_sha)
        if not args.appearance:row['unused_appearance_unchanged']=True
        report['epochs'].append(row);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    if updates!=2048 or digest(checkpoint)!=START_SHA or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(anchor_path)!=report['anchor_scores_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Budget or frozen evidence changed')
    if args.appearance and (digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(scores_path)!=report['control_snapshot']['selected_scores_sha256']):raise ValueError('Control changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');report['development_gate']=response_development_gate(baseline,selected,comparison);report['selected_checkpoint_sha256']=digest(out/f'epoch-{best_epoch}.pth');report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)

if __name__=='__main__':main()
