"""Fixed mean-token versus feature-token study; inference remains RGB-only."""
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
from reference_patch import initialize_patch_model,patch_forward,patch_development_gate
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from paired_training import grouped_batches
from pairwise_gate import require_completed_control,response_development_gate
from audit_consistent_reference import score_outputs
from training_checks import finite_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tokens',action='store_true');args=parser.parse_args()
    config=dict(seed=20261124,patch_init_seed=20261126,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.,reference_tokens=16)
    def seed():random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'multiref-control-v1/epoch-6.pth';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    if digest(checkpoint)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9' or digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c':raise ValueError('Changed fixed study inputs')
    cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];lookup={v['original_path']:data/v['prepared_path'] for v in prep['files']}
    if not prep['passed'] or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Incomplete study cohort')
    for item in prep['files']:
        if digest(lookup[item['original_path']])!=item['sha256']:raise ValueError('Changed prepared input')
    external_dir=base/'appearance-pairwise-replica-v1';external_path=external_dir/'results.json';external=load(external_path);external_scores=external_dir/f'epoch-{external["best_epoch"]}/scores.json';external_weight=external_dir/f'epoch-{external["best_epoch"]}.pth'
    if not external['finished'] or not external['appearance_enabled'] or external['best_epoch']!=7 or digest(external_path)!='3655c1120a7d854269304810c3be144e281d871a4cd6d1cd62c2236e2b7d093c' or digest(external_weight)!=external['selected_checkpoint_sha256']:raise ValueError('Changed external C10 benchmark')
    for name,sha in external['source_sha256'].items():
        if digest(external_dir/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed benchmark implementation')
    benchmark=load(external_scores);external_snapshot=dict(results_sha256=digest(external_path),scores_sha256=digest(external_scores),checkpoint_sha256=digest(external_weight))
    out=base/('reference-patch-tokens-v1' if args.tokens else 'reference-patch-mean-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=set(external['source_sha256'])|{'train_reference_patch.py','reference_patch.py','test_reference_patch.py','test_appearance_invariance.py','reference-patch-protocol.md','audit_consistent_reference.py'}
    for name in sorted(names):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Fixed feature-token development study, no semantic correspondence or independent acceptance',config=config,use_tokens=args.tokens,source_sha256={p.name:digest(p) for p in frozen.iterdir()},start_checkpoint_sha256=digest(checkpoint),cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),external_C10_snapshot=external_snapshot,initial_raw_equivalence=[],epochs=[],finished=False,app_promoted=False)
    mean_dir=base/'reference-patch-mean-v1';matched=None
    if args.tokens:
        mean=load(mean_dir/'results.json');require_completed_control(mean,config,report['source_sha256'])
        if mean['use_tokens'] or any(mean[k]!=report[k] for k in ['start_checkpoint_sha256','cohort_sha256','prepared_manifest_sha256','external_C10_snapshot']):raise ValueError('Wrong matched mean control')
        mp=mean_dir/f'epoch-{mean["best_epoch"]}/scores.json';mw=mean_dir/f'epoch-{mean["best_epoch"]}.pth'
        if digest(mw)!=mean['selected_checkpoint_sha256']:raise ValueError('Changed matched control weights')
        matched=load(mp);report['mean_control_snapshot']=dict(results_sha256=digest(mean_dir/'results.json'),scores_sha256=digest(mp),checkpoint_sha256=digest(mw),initial_sha256=digest(mean_dir/'epoch-0.pth'))
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(row,*[image(lookup[row[k]]) for k in ['content','reference','gt']]) for row in cohort['groups'][part]] for part in ['train','validation']};inputs=lambda v:(v.float()/255).cuda();truth={r['index']:gt.permute(1,2,0).numpy() for r,c,s,gt in samples['validation']}
    def pixels(value):return np.rint(finite_output(value,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    def save_scores(label,predictions):
        directory=out/label;directory.mkdir()
        for i,value in predictions.items():Image.fromarray(value).save(directory/f'{i}.png')
        scores=score_outputs(rows,predictions,truth)
        for row in scores['cases']:row['sha256']=digest(directory/f'{row["index"]}.png')
        (directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');return scores
    benchmark_model,benchmark_api,_=author_model();initialize_replication_model(benchmark_model,torch.load(checkpoint,map_location='cpu',weights_only=True));benchmark_model.load_state_dict(torch.load(external_weight,map_location='cpu',weights_only=True),strict=True);benchmark_model.eval().requires_grad_(False);predictions={}
    with torch.inference_mode():
        for row,c,s,gt in samples['validation']:
            i=row['index'];actual=pixels(appearance_forward(benchmark_model,benchmark_api,inputs(c[None]),inputs(s[None]))[0][0]);p=external_dir/f'epoch-{external["best_epoch"]}'/f'{i}.png'
            with Image.open(p) as im:saved=np.array(im.convert('RGB'))
            if not np.array_equal(actual,saved):raise ValueError('External C10 native output differs')
            predictions[i]=actual
    verified=score_outputs(rows,predictions,truth)
    for row in verified['cases']:row['sha256']=digest(external_dir/f'epoch-{external["best_epoch"]}'/f'{row["index"]}.png')
    if verified!=benchmark:raise ValueError('External C10 full metrics differ')
    report['external_C10_native']=dict(cases=64,maximum_code_difference=0,all_metrics_recomputed=True);del benchmark_model,benchmark_api
    model,api,author=author_model();base_state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(base_state,strict=True);model.eval();oracle=[];predictions={}
    with torch.inference_mode():
        for row,c,s,gt in samples['validation']:
            c,s=inputs(c[None]),inputs(s[None]);raw,grid=model(c,api['batch_histogram_1D'](api['srgb_to_ycbcr'](c)),api['batch_histogram_1D'](api['srgb_to_ycbcr'](s)));oracle.append((raw.cpu(),grid.cpu()));predictions[row['index']]=pixels(raw[0])
    baseline=save_scores('baseline-current',predictions)
    if baseline!=external['baseline']:raise ValueError('Current13 baseline differs')
    initialize_patch_model(model,base_state);report.update(author_revision=author['revision'],baseline=baseline,new_parameters=sum(v.numel() for v in model.reference_patch.parameters()))
    if report['new_parameters']!=65664 or any(not torch.equal(v.cpu(),model.state_dict()[k].cpu()) for k,v in base_state.items()):raise ValueError('Initial shared parameters differ')
    if args.tokens:
        initial=torch.load(mean_dir/'epoch-0.pth',map_location='cpu',weights_only=True)
        if set(initial)!=set(model.state_dict()) or any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):raise ValueError('Complete arm initial states differ')
        report['initial_complete_state_exact_to_mean']=True
    @torch.inference_mode()
    def evaluate(label):
        model.eval();predictions={}
        for number,(row,c,s,gt) in enumerate(samples['validation']):
            c,s=inputs(c[None]),inputs(s[None]);raw,grid=patch_forward(model,api,c,s,pooled=not args.tokens);predictions[row['index']]=pixels(raw[0])
            if label=='epoch-0':
                pairs=[(raw,grid),patch_forward(model,api,c,s,disabled=True),patch_forward(model,api,c,s,pooled=args.tokens)]
                difference=max(float((a.cpu()-oracle[number][0]).abs().max()) for a,b in pairs);difference=max(difference,max(float((b.cpu()-oracle[number][1]).abs().max()) for a,b in pairs))
                if difference!=0:raise ValueError('Initial enabled/disabled/opposite-mode raw/grid mismatch')
                report['initial_raw_equivalence'].append(dict(index=row['index'],maximum_float_difference=difference))
        return save_scores(label,predictions)
    initial=evaluate('epoch-0');oracle.clear()
    if initial!=baseline:raise ValueError('Initial enabled PNG/metrics differ')
    if args.tokens:(out/'epoch-0.pth').write_bytes((mean_dir/'epoch-0.pth').read_bytes())
    else:torch.save(model.state_dict(),out/'epoch-0.pth')
    report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(initial_raw_rgb_exact=True,external_native_exact=True,use_tokens=args.tokens,baseline=baseline['mean'])),flush=True)
    extra=list(model.reference_patch.parameters());identities={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in identities],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    seed();rng=np.random.default_rng(config['seed']);updates=0;best=baseline['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];order=[];maximum=dict(q=0.,key=0.,value=0.,projection=0.);start=time.perf_counter()
        for indices in grouped_batches(cohort['groups']['train'],rng):
            batch=[samples['train'][i] for i in indices];c,s,gt=[inputs(torch.stack([v[k] for v in batch])) for k in [1,2,3]];order.extend(v[0]['index'] for v in batch);optimizer.zero_grad(set_to_none=True);prediction=patch_forward(model,api,c,s,pooled=not args.tokens)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();head=model.reference_patch
            if head.kv.weight.grad is None:raise ValueError('Missing reference key/value gradients')
            for name,gradient in [('q',head.q.weight.grad),('key',head.kv.weight.grad[:128]),('value',head.kv.weight.grad[128:]),('projection',head.proj.weight.grad)]:
                if gradient is None or not torch.isfinite(gradient).all():raise ValueError('Missing/nonfinite branch gradients')
                maximum[name]=max(maximum[name],float(gradient.abs().max()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:print(json.dumps(dict(epoch=epoch,updates=updates,use_tokens=args.tokens,loss=float(loss))),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest()
        if len(order)!=1024 or len(set(order))!=1024 or order_sha!=external['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Training target order differs')
        if args.tokens and order_sha!=mean['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Arm target orders differ')
        torch.cuda.synchronize();seconds=time.perf_counter()-start;path=out/f'epoch-{epoch}.pth'
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained parameters')
        torch.save(model.state_dict(),path);scores=evaluate(f'epoch-{epoch}')
        if scores['mean']<best:best,best_epoch=scores['mean'],epoch
        item=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=scores['mean'],delta_mean=scores['delta_mean'],checkpoint_sha256=digest(path),target_order_sha256=order_sha,branch_gradient_maximum=maximum);report['epochs'].append(item);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)
    if updates!=2048 or digest(checkpoint)!=report['start_checkpoint_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()) or digest(external_path)!=external_snapshot['results_sha256'] or digest(external_scores)!=external_snapshot['scores_sha256'] or digest(external_weight)!=external_snapshot['checkpoint_sha256']:raise ValueError('Frozen source/data/benchmark or training budget changed')
    if args.tokens and (digest(mean_dir/'results.json')!=report['mean_control_snapshot']['results_sha256'] or digest(mp)!=report['mean_control_snapshot']['scores_sha256'] or digest(mw)!=report['mean_control_snapshot']['checkpoint_sha256'] or digest(mean_dir/'epoch-0.pth')!=report['mean_control_snapshot']['initial_sha256']):raise ValueError('Matched control changed during candidate run')
    selected=load(out/f'epoch-{best_epoch}/scores.json');qualification=patch_development_gate(baseline,selected,matched,benchmark) if args.tokens else dict(current_only=response_development_gate(baseline,selected),external_C10=response_development_gate(baseline,selected,benchmark),proceed_to_independent_evaluation=False)
    report.update(development_gate=qualification,selected_checkpoint_sha256=digest(out/f'epoch-{best_epoch}.pth'),finished=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(qualification),flush=True)


if __name__=='__main__':main()
