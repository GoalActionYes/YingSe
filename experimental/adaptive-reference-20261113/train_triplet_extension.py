"""Frozen continuation experiment on existing development data, never reserved."""
import json
import argparse
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output
from training_gates import development_gate

START_SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--joint',action='store_true');parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    config=dict(seed=20261115 if args.expanded else 20261114,lr=1e-5,weight_decay=1e-4,batch=4,extra_epochs=8 if args.expanded else 24,max_updates=1024 if args.expanded else 768,
        start_epoch=8,evaluate_epochs=list(range(9,17)) if args.expanded else [12,16,20,24,28,32],gradient_weight=.1,gradient_clip=1.,optimizer_state='reset AdamW')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/('cohort-v2' if args.expanded else 'cohort-v1');checkpoint=base/'training-v1/epoch-8.pth'
    if digest(checkpoint)!=START_SHA:raise ValueError('Changed starting checkpoint')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));prep_path=data/'srgb-v1/manifest.json';prep=json.loads(prep_path.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(data/'cohort.json'):raise ValueError('Invalid prepared cohort')
    lookup={entry['original_path']:data/entry['prepared_path'] for entry in prep['files']}
    for entry in prep['files']:
        if digest(lookup[entry['original_path']])!=entry['sha256']:raise ValueError('Changed prepared file')
    run_name=('training-expanded-joint-v1' if args.joint else 'training-expanded-control-v1') if args.expanded else ('training-joint-v1' if args.joint else 'training-v2')
    control_dir=base/('training-expanded-control-v1' if args.expanded else 'training-v2')
    out=base/run_name;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['train_triplet_extension.py','triplet-extension-protocol.md','train_triplet_pilot.py','training_checks.py','training_gates.py']
    if args.expanded:names+=['expanded-triplet-protocol.md']
    if args.joint:names+=['joint_features.py','joint-feature-protocol.md']
    for name in names:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(config=config,joint_features=args.joint,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(prep_path),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},scope='Development only; no reserved data, no app replacement',epochs=[],finished=False)
    if args.joint:
        control_path=control_dir/'results.json';control=json.loads(control_path.read_text(encoding='utf-8'))
        if (not control['finished'] or control['config']!=config or control['cohort_sha256']!=report['cohort_sha256']
                or control['prepared_manifest_sha256']!=report['prepared_manifest_sha256'] or control['start_checkpoint_sha256']!=START_SHA):raise ValueError('Mismatched continuation control')
        control_scores_path=control_dir/f'epoch-{control["best_epoch"]}/scores.json'
        report['control_snapshot']=dict(results_sha256=digest(control_path),scores_sha256=digest(control_scores_path),best_epoch=control['best_epoch'])
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,author=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite start weights')
    model.load_state_dict(state,strict=True);samples={}
    for part in ['train','validation']:
        samples[part]=[]
        for row in cohort['groups'][part]:
            tensors=[]
            for role in ['content','reference','gt']:
                with Image.open(lookup[row[role]]) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                tensors.append(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
            samples[part].append((row,*tensors))
    if len(samples['train'])!=(512 if args.expanded else 128) or len(samples['validation'])!=(64 if args.expanded else 32):raise ValueError('Unexpected sample count')
    oracle=[]
    if args.joint:
        from joint_features import enable_joint_input,joint_forward
        model.eval()
        with torch.inference_mode():
            for row,c,s,gt in samples['validation']:
                c,s=c[None].cuda(),s[None].cuda();hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
                oracle.append(finite_output(model(c,hc,hs)[0],c.shape)[0].cpu())
        enable_joint_input(model);torch.save(model.state_dict(),out/'epoch-8.pth')
        report['initial_checkpoint_sha256']=digest(out/'epoch-8.pth')
        report['initial_equivalence']=[]
    def forward(c,s):
        if args.joint:return joint_forward(model,api,c,s)[0].clamp(0,1)
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        raw,grid=model(c,hc,hs);finite_output(grid,(c.shape[0],13,8,16,16))
        return finite_output(raw,c.shape).clamp(0,1)
    def pixels(value):return np.rint(finite_output(value,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();folder=out/f'epoch-{epoch}';folder.mkdir();rows=[]
        for index,(row,c,s,gt) in enumerate(samples['validation']):
            if args.joint and epoch==8:
                raw=joint_forward(model,api,c[None].cuda(),s[None].cuda())[0][0]
                delta=float((raw.cpu()-oracle[index]).abs().max());prediction=pixels(raw)
                reference=pixels(oracle[index]);byte_delta=int(np.abs(prediction.astype(int)-reference.astype(int)).max())
                report['initial_equivalence'].append(dict(index=row['index'],max_float_delta=delta,max_rgb8_delta=byte_delta,changed_values=int(np.count_nonzero(prediction!=reference))))
                if delta>1e-5 or byte_delta>1:raise ValueError('Joint input changed initial model beyond predefined tolerance')
            else:prediction=pixels(forward(c[None].cuda(),s[None].cuda())[0])
            path=folder/f'{row["index"]}.png';Image.fromarray(prediction).save(path)
            rows.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-pixels(gt)).mean()),input_mae=float(np.abs(pixels(c).astype(float)-pixels(gt)).mean()),sha256=digest(path)))
        result=dict(mean=float(np.mean([r['mae'] for r in rows])),input_mean=float(np.mean([r['input_mae'] for r in rows])),cases=rows)
        if args.expanded:result['subgroups']=dict(original32=float(np.mean([r['mae'] for r in rows[:32]])),added32=float(np.mean([r['mae'] for r in rows[32:]])))
        (folder/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    baseline=evaluate(8);best=baseline['mean'];best_epoch=8;report.update(baseline=baseline,author=author)
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'baseline':best}),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);rng=np.random.default_rng(config['seed']);updates=0
    final_epoch=config['start_epoch']+config['extra_epochs']
    for epoch in range(9,final_epoch+1):
        model.train();start=time.perf_counter();losses=[];order=rng.permutation(len(samples['train']))
        for offset in range(0,len(order),4):
            chosen=[samples['train'][int(j)] for j in order[offset:offset+4]]
            c,s,gt=[torch.stack([r[k] for r in chosen]).cuda() for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);pred=forward(c,s)
            loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained weights')
        path=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),path)
        record=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,checkpoint_sha256=digest(path))
        if epoch in config['evaluate_epochs']:
            validation=evaluate(epoch);record['validation_mean']=validation['mean']
            if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best,finished=epoch==final_epoch)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if updates!=config['max_updates']:raise ValueError('Unexpected update count')
    selected=json.loads((out/f'epoch-{best_epoch}/scores.json').read_text(encoding='utf-8'))
    control_scores=None
    if args.joint:
        if digest(control_path)!=report['control_snapshot']['results_sha256'] or digest(control_scores_path)!=report['control_snapshot']['scores_sha256']:raise ValueError('Control changed during training')
        control_scores=json.loads(control_scores_path.read_text(encoding='utf-8'))
    gate=development_gate(baseline,selected,args.expanded,control_scores)
    report['development_gate']=gate;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
