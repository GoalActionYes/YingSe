"""Two fixed equal-update continuations; run control first, then --paired."""
import argparse
from collections import Counter
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
from coverage_training import block_batches
from staged_response import staged_loss,staged_gate
COHORT_SHA='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate,require_completed_control


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';old_data=base/'cohort-multiref-v1';prior=base/'appearance-pairwise-replica-v1';warm=base/'training-coverage-expanded-v2';warm_state=warm/'epoch-4.pth';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort=load(data/'cohort.json');old_cohort=load(old_data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');old_prep=load(old_data/'srgb-v1/manifest.json');original=load(prior/'results.json')
    if digest(data/'cohort.json')!=COHORT_SHA or digest(old_data/'cohort.json')!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(old_data/'srgb-v1/manifest.json')!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c':raise ValueError('Data definition changed')
    if prep['cohort_sha256']!=COHORT_SHA or not prep['passed'] or any(prep[k] for k in ('invalid_pairs','cross_split_exact_duplicates','cross_split_near_duplicates')):raise ValueError('Preparation/duplicate screen has not passed; manual flags need separate audited acceptance')
    if cohort['groups']['validation']!=old_cohort['groups']['validation'] or len(cohort['groups']['train'])!=4096 or len(old_cohort['groups']['train'])!=1024:raise ValueError('Data counts/dev changed')
    if digest(prior/'results.json')!='3655c1120a7d854269304810c3be144e281d871a4cd6d1cd62c2236e2b7d093c' or digest(prior/'epoch-7.pth')!='abd4757eff1c5133101732c1f04a051ce29400358282baacf7a43030a5584e38':raise ValueError('Common C10 initialization changed')
    for name,sha in original['source_sha256'].items():
        if digest(prior/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Original C10 source changed')
    lookup={v['original_path']:v for v in prep['files']};old_lookup={v['original_path']:v for v in old_prep['files']}
    for entry in prep['files']:
        if not (data/entry['prepared_path']).resolve().is_relative_to(base.resolve()):raise ValueError('Shared prepared path escaped dataset root')
        if digest(data/entry['prepared_path'])!=entry['sha256']:raise ValueError('Prepared file changed')
    for name,sha in prep['protected_inputs_sha256'].items():
        if digest(base/name)!=sha:raise ValueError('Revision preparation input changed')
    for name,sha in prep['source_sha256'].items():
        if digest(data/'srgb-v1/frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Preparation source changed')
    for part in ('train','validation'):
        for row in old_cohort['groups'][part]:
            for role in ('content','reference','gt'):
                if lookup[row[role]]['pixel_sha256']!=old_lookup[row[role]]['pixel_sha256']:raise ValueError('Original development/training pixels changed')
    if digest(warm/'results.json')!='35e2e847470217fa68fa8829f1607971b80da6124bba580effec64846bb9d3d8' or digest(warm_state)!='cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1':raise ValueError('Staged warm start changed')
    warm_record=load(warm/'results.json')
    for name,sha in warm_record['source_sha256'].items():
        if digest(warm/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Warm source changed')
    config=dict(seed=20261124,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=4,max_updates=4096,delta_weight=.25,gradient_weight=.1,gradient_clip=1.)
    control_dir=base/'staged-response-control-v1';control=None;comparison=None
    if args.paired:
        control=load(control_dir/'results.json');require_completed_control(control,config,control['source_sha256'])
        if control['paired'] or control['passes_per_block']!=1 or control['cohort_sha256']!=COHORT_SHA or control['prepared_manifest_sha256']!=digest(data/'srgb-v1/manifest.json'):raise ValueError('Wrong continuation control')
        comparison=load(control_dir/f'epoch-{control["best_epoch"]}/scores.json')
        if digest(control_dir/f'epoch-{control["best_epoch"]}.pth')!=control['selected_checkpoint_sha256']:raise ValueError('Control selected weights changed')
    out=base/('staged-response-paired-v1' if args.paired else 'staged-response-control-v1');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=set(original['source_sha256'])|{'train_staged_response.py','staged_response.py','paired_training.py','staged-response-trial-protocol.md','train_training_coverage_v2.py','coverage_training.py','coverage_asset_plan.py','prepare_training_coverage.py','training_coverage.py','prepare_triplet_cohort.py','training-coverage-trial-protocol.md','coverage-trial-v2-binding.md','audit_consistent_reference.py'}
    for name in sorted(names):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sources={p.name:digest(p) for p in frozen.iterdir()}
    if control and control['source_sha256']!=sources:raise ValueError('Control implementation changed')
    report=dict(data_revision='capture-alias-v2',config=config,paired=args.paired,passes_per_block=1,source_sha256=sources,cohort_sha256=COHORT_SHA,prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),old_cohort_sha256=digest(old_data/'cohort.json'),start_checkpoint_sha256=digest(warm_state),original_results_sha256=digest(prior/'results.json'),epochs=[],finished=False,app_promoted=False,baseline=original['baseline'],scope='Staged4096update paired-response vs plain continuation, no independent/product acceptance')
    if control:report['control_snapshot']=dict(results_sha256=digest(control_dir/'results.json'),selected_checkpoint_sha256=control['selected_checkpoint_sha256'],scores_sha256=digest(control_dir/f'epoch-{control["best_epoch"]}/scores.json'))
    protected_paths=[data/'cohort.json',data/'srgb-v1/manifest.json',old_data/'cohort.json',old_data/'srgb-v1/manifest.json',prior/'results.json',prior/'epoch-7.pth',prior/'epoch-7/scores.json',warm/'results.json',warm_state,warm/'epoch-4/scores.json']
    if control:protected_paths += [control_dir/'results.json',control_dir/f'epoch-{control["best_epoch"]}.pth',control_dir/f'epoch-{control["best_epoch"]}/scores.json']
    report['protected_inputs_sha256']={str(p.relative_to(base)):digest(p) for p in protected_paths}
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def seed():random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();model,api,author=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));initial=torch.load(warm_state,map_location='cpu',weights_only=True);model.load_state_dict(initial,strict=True)
    if any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()) or set(initial)!=set(model.state_dict()):raise ValueError('Complete common state differs')
    report.update(initial_complete_state_exact=True,author_revision=author['revision'],initial_native_raw_grid=[]);(out/'epoch-0.pth').write_bytes(warm_state.read_bytes())
    native_ref,native_api,_=author_model();initialize_replication_model(native_ref,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));native_ref.load_state_dict(initial,strict=True);native_ref.eval()
    rows=cohort['groups']['validation'];train=cohort['groups']['train'];asset_paths=sorted({r[k] for r in rows+train for k in ('content','reference','gt')});cache={}
    for n,path in enumerate(asset_paths,1):
        with Image.open(data/lookup[path]['prepared_path']) as im:cache[path]=torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
        if n%1000==0:print('loaded training assets',n,'/',len(asset_paths),flush=True)
    truth={r['index']:cache[r['gt']].permute(1,2,0).numpy() for r in rows};inputs=lambda a:(a.float()/255).cuda()
    @torch.inference_mode()
    def evaluate(epoch):
        model.eval();directory=out/f'epoch-{epoch}';directory.mkdir();predictions={}
        for row in rows:
            c,s=[inputs(cache[row[k]][None]) for k in ('content','reference')];raw,grid=appearance_forward(model,api,c,s)
            if not torch.isfinite(raw).all() or not torch.isfinite(grid).all():raise ValueError('Nonfinite output')
            if epoch==0:
                expected,expected_grid=appearance_forward(native_ref,native_api,c,s);difference=max(float((raw-expected).abs().max()),float((grid-expected_grid).abs().max()))
                if difference!=0:raise ValueError('Independent staged warm-start raw/grid differs')
                report['initial_native_raw_grid'].append(dict(index=row['index'],maximum_float_difference=difference))
            pixels=np.rint(raw[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8);predictions[row['index']]=pixels;Image.fromarray(pixels).save(directory/f'{row["index"]}.png')
        scores=score_outputs(rows,predictions,truth)
        for case in scores['cases']:case['sha256']=digest(directory/f'{case["index"]}.png')
        (directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');return scores
    start=evaluate(0);old_scores=load(prior/'epoch-7/scores.json')
    if start!=load(warm/'epoch-4/scores.json') or control and start!=control['warm_start_scores']:raise ValueError('Native staged warm-start initial64 outputs/full metrics differ')
    native_ref=None;torch.cuda.empty_cache()
    report['warm_start_scores']=start;report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'initial-bridge.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());ids={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],weight_decay=config['weight_decay']);seed();rng=np.random.default_rng(config['seed']);updates=0;best=start['mean'];best_epoch=0
    for epoch in range(1,5):
        model.train();losses=[];term_values={k:[] for k in ('rgb','edge','delta')};order=[];started=time.perf_counter();batches=block_batches(train,rng,report['passes_per_block'])
        if len(batches)!=1024:raise ValueError('Unequal update budget')
        for indices in batches:
            batch=[train[i] for i in indices];c,s,gt=[inputs(torch.stack([cache[r[k]] for r in batch])) for k in ('content','reference','gt')];order.extend(r['index'] for r in batch);optimizer.zero_grad(set_to_none=True);prediction=appearance_forward(model,api,c,s)[0].clamp(0,1)
            terms=staged_loss(prediction,gt,args.paired);loss=terms['total']
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            for name in term_values:term_values[name].append(float(terms[name]))
            if updates%256==0:print(json.dumps(dict(epoch=epoch,updates=updates,loss=float(loss))),flush=True)
        if len(order)!=4096 or set(order)!={r['index'] for r in train} or set(Counter(order).values())!={report['passes_per_block']}:raise ValueError('Exposure budget differs')
        torch.cuda.synchronize();seconds=time.perf_counter()-started;path=out/f'epoch-{epoch}.pth'
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained state')
        torch.save(model.state_dict(),path);scores=evaluate(epoch)
        if scores['mean']<best:best,best_epoch=scores['mean'],epoch
        item=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),term_means={k:float(np.mean(v)) for k,v in term_values.items()},training_seconds=seconds,validation_mean=scores['mean'],delta_mean=scores['delta_mean'],checkpoint_sha256=digest(path),target_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest());report['epochs'].append(item);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)
    if updates!=4096 or digest(data/'cohort.json')!=COHORT_SHA or digest(data/'srgb-v1/manifest.json')!=report['prepared_manifest_sha256'] or digest(warm_state)!=report['start_checkpoint_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in sources.items()):raise ValueError('Frozen experiment changed')
    if any(digest(base/n)!=sha for n,sha in report['protected_inputs_sha256'].items()):raise ValueError('Protected input/score changed midrun')
    if control and digest(control_dir/'results.json')!=report['control_snapshot']['results_sha256']:raise ValueError('Matched control changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');original_gate=response_development_gate(report['baseline'],selected,old_scores);matched_gate=response_development_gate(report['baseline'],selected,comparison) if control else original_gate
    final_gate=staged_gate(matched_gate,original_gate,args.paired)
    report.update(development_gate=final_gate,matched_control_gate=matched_gate,original_C10_gate=original_gate,proceed_to_independent_evaluation=final_gate['proceed_to_independent_evaluation'],selected_checkpoint_sha256=digest(out/f'epoch-{best_epoch}.pth'),finished=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(selected=best_epoch,mean=best,matched_gate=matched_gate,original_C10_gate=original_gate)),flush=True)


if __name__=='__main__':main()
