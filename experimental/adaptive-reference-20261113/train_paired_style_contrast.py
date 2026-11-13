"""One fixed contrastive candidate with reused same-architecture C10 control."""
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
from paired_style_contrast import paired_style_contrast
from paired_training import grouped_batches
from pairwise_gate import require_completed_control,response_development_gate
from multiref_response import response_metrics
from tone_response import tone_response
from training_checks import finite_output


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';control_dir=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    control_path=control_dir/'results.json';control=load(control_path)
    base_config=dict(seed=20261124,appearance_init_seed=20261125,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    require_completed_control(control,base_config,control['source_sha256'])
    if not control['appearance_enabled']:raise ValueError('Expected stronger appearance-enabled C10 control')
    for name,sha in control['source_sha256'].items():
        if digest(control_dir/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed completed control source')
    scores_path=control_dir/f'epoch-{control["best_epoch"]}/scores.json';comparison=load(scores_path);initial_path=control_dir/'epoch-0.pth';selected_path=control_dir/f'epoch-{control["best_epoch"]}.pth'
    if digest(initial_path)!=control['initial_checkpoint_sha256'] or digest(selected_path)!=control['selected_checkpoint_sha256']:raise ValueError('Changed reused weights')
    snapshot=dict(results_sha256=digest(control_path),scores_sha256=digest(scores_path),initial_sha256=digest(initial_path),selected_sha256=digest(selected_path))
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(cohort_path)!=control['cohort_sha256'] or digest(prep_path)!=control['prepared_manifest_sha256'] or digest(checkpoint)!=control['start_checkpoint_sha256'] or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Changed data/base')
    lookup={v['original_path']:data/v['prepared_path'] for v in prep['files']}
    for row in prep['files']:
        if digest(lookup[row['original_path']])!=row['sha256']:raise ValueError('Changed prepared file')
    diagnostic=base/'real-style-correspondence-v1/results.json';diag=load(diagnostic)
    if not diag['finished'] or any(diag[k]!=control[k] for k in ['cohort_sha256','prepared_manifest_sha256']) or diag['checkpoint_sha256']!=control['start_checkpoint_sha256']:raise ValueError('Diagnostic scope differs')
    config=dict(base_config,style_weight=.01,temperature=.1);out=base/'paired-style-contrast-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in sorted(set(control['source_sha256'])|{'train_paired_style_contrast.py','paired_style_contrast.py','test_paired_style_contrast.py','paired-style-training-protocol.md'}):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='One real-pair contrastive training candidate on consumed development; no independent acceptance',config=config,control_snapshot=snapshot,diagnostic_results_sha256=digest(diagnostic),source_sha256={p.name:digest(p) for p in frozen.iterdir()},cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),start_checkpoint_sha256=digest(checkpoint),epochs=[],initial_raw_equivalence=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def seed():random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();model,api,author=author_model();initialize_replication_model(model,torch.load(checkpoint,map_location='cpu',weights_only=True));initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    if set(initial)!=set(model.state_dict()) or any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('Full initial model differs')
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) or (isinstance(m,torch.nn.Dropout) and m.p!=0) for m in model.modules()):raise ValueError('Additional encoding would have stochastic/stateful effects')
    report.update(initial_complete_state_exact_to_C10=True,author_revision=author['revision'])
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(r,*[image(lookup[r[k]]) for k in ['content','reference','gt']]) for r in cohort['groups'][part]] for part in ['train','validation']};inputs=lambda v:(v.float()/255).cuda()
    pixels=lambda raw:np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(label):
        model.eval();directory=out/label;directory.mkdir();cases=[];groups={}
        for row,c,s,gt in samples['validation']:
            c,s=inputs(c[None]),inputs(s[None]);raw,grid=appearance_forward(model,api,c,s);prediction=pixels(raw[0]);truth=gt.permute(1,2,0).numpy();path=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            if label=='epoch-0':
                direct,direct_grid=model(c,api['batch_histogram_1D'](api['srgb_to_ycbcr'](c)),api['batch_histogram_1D'](api['srgb_to_ycbcr'](s)));difference=max(float((raw-direct).abs().max()),float((grid-direct_grid).abs().max()))
                if difference!=0:raise ValueError('Initial current13 raw/grid mismatch')
                report['initial_raw_equivalence'].append(dict(index=row['index'],maximum_float_difference=difference))
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(path)));g=groups.setdefault(row['group_id'],{})
            if row['style_index'] in g:raise ValueError('Duplicate validation style')
            g[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete paired validation')
        pairs=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        scores=dict(mean=float(np.mean([v['mae'] for v in cases])),cases=cases,paired=pairs,delta_mean=float(np.mean([v['delta_mae'] for v in pairs])));(directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');return scores
    model.load_state_dict(torch.load(selected_path,map_location='cpu',weights_only=True),strict=True)
    if evaluate('reused-control')!=comparison:raise ValueError('Selected C10 native outputs/metrics differ')
    model.load_state_dict(initial,strict=True)
    if any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('Failed to restore complete initial state')
    baseline=evaluate('epoch-0')
    if baseline!=control['baseline']:raise ValueError('Initial/current13 RGB scores differ')
    report['baseline']=baseline;(out/'epoch-0.pth').write_bytes(initial_path.read_bytes());report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(initial_raw_rgb_exact=True,control_native_exact=True,baseline=baseline['mean'])),flush=True)
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());identities={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    seed();rng=np.random.default_rng(config['seed']);updates=0;best=baseline['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];rgb_losses=[];gradient_losses=[];style_losses=[];order=[];start=time.perf_counter()
        for indices in grouped_batches(cohort['groups']['train'],rng):
            batch=[samples['train'][i] for i in indices]
            for a,b in [(batch[0][0],batch[1][0]),(batch[2][0],batch[3][0])]:
                if a['group_id']!=b['group_id'] or a['content']!=b['content'] or a['style_index']==b['style_index']:raise ValueError('Wrong contrastive positives/negatives')
            c,s,gt=[inputs(torch.stack([r[k] for r in batch])) for k in [1,2,3]];order.extend(r[0]['index'] for r in batch);optimizer.zero_grad(set_to_none=True);prediction=appearance_forward(model,api,c,s)[0].clamp(0,1)
            rgb_loss=F.l1_loss(prediction,gt);gradient=F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1])
            ref_features=model.img_projection(api['TF'].resize(s,(64,64),antialias=True)).mean((-2,-1));gt_features=model.img_projection(api['TF'].resize(gt,(64,64),antialias=True)).mean((-2,-1));style=paired_style_contrast(ref_features,gt_features,config['temperature']);loss=rgb_loss+.1*gradient+config['style_weight']*style
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            if updates==0:
                gradients=torch.autograd.grad(style,list(model.img_projection.parameters()),retain_graph=True);norm=float(torch.stack([v.square().sum() for v in gradients]).sum().sqrt())
                if not np.isfinite(norm) or norm<=0:raise ValueError('Auxiliary encoder gradient missing')
                report['initial_auxiliary_encoder_gradient_norm']=norm
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss));rgb_losses.append(float(rgb_loss));gradient_losses.append(float(gradient));style_losses.append(float(style))
            if updates%64==0:print(json.dumps(dict(epoch=epoch,updates=updates,loss=float(loss),style_ce=float(style))),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if len(order)!=1024 or len(set(order))!=1024 or order_sha!=control['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Target order differs from C10')
        torch.cuda.synchronize();seconds=time.perf_counter()-start;path=out/f'epoch-{epoch}.pth'
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained state')
        torch.save(model.state_dict(),path);scores=evaluate(f'epoch-{epoch}')
        if scores['mean']<best:best,best_epoch=scores['mean'],epoch
        item=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),rgb_l1=float(np.mean(rgb_losses)),gradient_l1=float(np.mean(gradient_losses)),style_ce=float(np.mean(style_losses)),training_seconds=seconds,validation_mean=scores['mean'],delta_mean=scores['delta_mean'],checkpoint_sha256=digest(path),target_order_sha256=order_sha)
        report['epochs'].append(item);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)
    if updates!=2048 or digest(control_path)!=snapshot['results_sha256'] or digest(scores_path)!=snapshot['scores_sha256'] or digest(initial_path)!=snapshot['initial_sha256'] or digest(selected_path)!=snapshot['selected_sha256'] or digest(diagnostic)!=report['diagnostic_results_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(checkpoint)!=report['start_checkpoint_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Budget or frozen evidence changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');report.update(development_gate=response_development_gate(baseline,selected,comparison),selected_checkpoint_sha256=digest(out/f'epoch-{best_epoch}.pth'),finished=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)


if __name__=='__main__':main()
