"""Validate reusing the completed real-reference control, without retraining it."""
import json
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from train_consistent_reference import forward as control_forward
from pairwise_context import enable_pairwise_context
from appearance_pairwise import enable_appearance_context,appearance_forward
from pairwise_gate import require_completed_control
from paired_training import grouped_batches


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="appearance-pairwise-bridge-v1");parser.add_argument("--step-device",choices=["cpu","cuda"],default="cuda");args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in (".",".."):raise ValueError("Expected run directory name")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';control_dir=base/'consistent-reference-control-v1';data=base/'cohort-multiref-v1'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));control_path=control_dir/'results.json';control=load(control_path)
    require_completed_control(control,control['config'],control['source_sha256'])
    if control['augment_reference'] or control['config']!=dict(seed=20261122,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.):raise ValueError('Unexpected reused control')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path)
    if digest(cohort_path)!=control['cohort_sha256'] or digest(prep_path)!=control['prepared_manifest_sha256'] or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0) or not prep['passed']:raise ValueError('Changed common data')
    for name,sha in control['source_sha256'].items():
        if digest(control_dir/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed control implementation')
    initial_path=control_dir/'epoch-0.pth';selected_path=control_dir/f'epoch-{control["best_epoch"]}.pth';scores_path=control_dir/f'epoch-{control["best_epoch"]}/scores.json'
    if digest(initial_path)!=control['initial_checkpoint_sha256'] or digest(selected_path)!=control['selected_checkpoint_sha256']:raise ValueError('Changed reused weights')
    out=base/args.run;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=sorted(set(['bridge_appearance_control.py','appearance_pairwise.py','reference_context.py','appearance-pairwise-protocol.md',*control['source_sha256']]))
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Control reuse bridge, not replay of historical2048 updates or independent acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),control_snapshot=dict(results_sha256=digest(control_path),selected_scores_sha256=digest(scores_path),initial_checkpoint_sha256=digest(initial_path),selected_checkpoint_sha256=digest(selected_path)),initial_native=[],selected_control_native=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lookup={r['original_path']:r for r in prep['files']}
    def read(relative):
        item=lookup[relative];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('Changed prepared input')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def tensor(rgb):return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    def pixels(raw):return np.rint(raw[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    old,api,_=author_model();new,api2,_=author_model();enable_pairwise_context(old);enable_pairwise_context(new)
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    old.load_state_dict(initial,strict=True);new.load_state_dict(initial,strict=True)
    torch.manual_seed(20261123);torch.cuda.manual_seed_all(20261123);enable_appearance_context(new)
    added={k for k in new.state_dict() if k not in initial}
    if added!={f'reference_adapter.layers.{i}.{field}' for i in [0,2] for field in ['weight','bias']}:raise ValueError('Unexpected new state entries')
    if any(not torch.equal(v,new.state_dict()[k].cpu()) for k,v in initial.items()):raise ValueError('Shared initial parameters differ')
    new_initial={k:v.detach().cpu().clone() for k,v in new.state_dict().items()}
    torch.save(new_initial,out/'epoch-0.pth');report['candidate_initial_checkpoint_sha256']=digest(out/'epoch-0.pth');report['shared_initial_parameters_identical']=True;report['new_parameter_count']=sum(p.numel() for p in new.reference_adapter.parameters())
    base_weights=torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True)
    if digest(base/'multiref-control-v1/epoch-6.pth')!=control['start_checkpoint_sha256'] or any(not torch.equal(v,initial[k]) for k,v in base_weights.items()):raise ValueError('Initial base differs from current13')
    baseline_scores={r['index']:r for r in control['baseline']['cases']};samples=[(r,read(r['content']),read(r['reference'])) for r in cohort['groups']['validation']]
    old.eval();new.eval()
    with torch.inference_mode():
        for row,c,s in samples:
            c,s=tensor(c),tensor(s);a,ga=control_forward(old,api,c,s);b,gb=appearance_forward(new,api2,c,s)
            delta=max(float((a-b).abs().max()),float((ga-gb).abs().max()))
            if delta!=0:raise ValueError('Zero appearance head changes initial raw/grid')
            actual=pixels(b);p=control_dir/'baseline-current'/f'{row["index"]}.png'
            if digest(p)!=baseline_scores[row['index']]['sha256']:raise ValueError('Changed current baseline PNG')
            with Image.open(p) as im:saved=np.array(im.convert('RGB'))
            code=int(np.abs(actual.astype(int)-saved.astype(int)).max())
            if code!=0:raise ValueError('Initial output differs from current13')
            report['initial_native'].append(dict(index=row['index'],raw_grid_maximum_difference=delta,maximum_code_difference=code))
    config=control['config'];indices=grouped_batches(cohort['groups']['train'],np.random.default_rng(config['seed']))[0];batch=[cohort['groups']['train'][i] for i in indices]
    c,s,gt=[torch.cat([tensor(read(r[role])) for r in batch]) for role in ['content','reference','gt']]
    def optimizer(model,appearance):
        extra=list(model.pairwise_adapter.parameters())+(list(model.reference_adapter.parameters()) if appearance else []);ids={id(p) for p in extra}
        return torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    if args.step_device=='cpu':
        old.cpu();new.cpu();c,s,gt=c.cpu(),s.cpu(),gt.cpu()
    report['step_device']=args.step_device
    report['pre_step_shared_difference']=max(float((v-new.state_dict()[k]).abs().max()) for k,v in old.state_dict().items())
    step=[]
    for model,functions,opt in [(old,lambda:control_forward(old,api,c,s)[0],optimizer(old,False)),(new,lambda:appearance_forward(new,api2,c,s,disable_appearance=True)[0],optimizer(new,True))]:
        model.train();random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);opt.zero_grad(set_to_none=True);prediction=functions().clamp(0,1)
        loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();step.append(dict(loss=float(loss),gradient_norm=float(norm)))
    difference=max(float((v-new.state_dict()[k]).abs().max()) for k,v in old.state_dict().items())
    mismatches=sorted([(k,float((v-new.state_dict()[k]).abs().max())) for k,v in old.state_dict().items()],key=lambda x:x[1],reverse=True)[:12]
    diagnostic=dict(pre_step_shared_difference=report['pre_step_shared_difference'],step=step,shared_update_difference=difference,largest_shared_differences=mismatches,new_params_with_gradient=sum(p.grad is not None for p in new.reference_adapter.parameters()))
    (out/'step-diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8');print(json.dumps(diagnostic),flush=True)
    if step[0]!=step[1] or difference!=0 or diagnostic['new_params_with_gradient']!=0:raise ValueError('Disabled branch changes shared training update')
    report['one_step_bridge']=dict(device=args.step_device,indices=[r['index'] for r in batch],control=step[0],disabled_appearance=step[1],shared_parameter_maximum_difference=difference,new_parameter_gradients_absent=True)
    old.cuda();new.cuda()
    selected=torch.load(selected_path,map_location='cpu',weights_only=True);old.load_state_dict(selected,strict=True)
    new.load_state_dict(new_initial,strict=True);missing=new.load_state_dict(selected,strict=False)
    if set(missing.missing_keys)!=added or missing.unexpected_keys:raise ValueError('Selected control state mismatch')
    selected_scores={r['index']:r for r in load(scores_path)['cases']};old.eval();new.eval()
    with torch.inference_mode():
        for row,c,s in samples:
            c,s=tensor(c),tensor(s);a,ga=control_forward(old,api,c,s);b,gb=appearance_forward(new,api2,c,s,disable_appearance=True)
            delta=max(float((a-b).abs().max()),float((ga-gb).abs().max()))
            p=control_dir/f'epoch-{control["best_epoch"]}'/f'{row["index"]}.png'
            if digest(p)!=selected_scores[row['index']]['sha256']:raise ValueError('Changed selected control PNG')
            with Image.open(p) as im:saved=np.array(im.convert('RGB'))
            code=int(np.abs(pixels(b).astype(int)-saved.astype(int)).max())
            if delta!=0 or code!=0:raise ValueError('Selected control bridge differs')
            report['selected_control_native'].append(dict(index=row['index'],raw_grid_maximum_difference=delta,maximum_code_difference=code))
    if digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(scores_path)!=report['control_snapshot']['selected_scores_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Frozen bridge evidence changed')
    report.update(finished=True,passed=True,app_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,passed=True,new_parameters=report['new_parameter_count'],one_step=report['one_step_bridge'],initial_native_cases=len(report['initial_native']),selected_control_native_cases=len(report['selected_control_native']))),flush=True)

if __name__=='__main__':main()
