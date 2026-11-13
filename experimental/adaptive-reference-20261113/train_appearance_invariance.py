"""Complete the two missing cells of the fixed appearance/augmentation study."""
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
from appearance_invariance import invariance_gate
from consistent_reference import use_synthetic_reference
from pairwise_gate import require_completed_control
from paired_training import grouped_batches
from training_checks import finite_output
from multiref_response import response_metrics
from tone_response import tone_response

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--appearance',action='store_true');args=parser.parse_args();cell='C11' if args.appearance else 'C01'
    config=dict(seed=20261124,appearance_init_seed=20261125,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=8,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    def seed():
        random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    seed();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';checkpoint=base/'multiref-control-v1/epoch-6.pth';cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';generated_dir=base/'cohort-consistent-reference-v1';generated_path=generated_dir/'results.json'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    if digest(checkpoint)!=START_SHA or digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c':raise ValueError('Changed fixed inputs')
    cohort,prep,generated=load(cohort_path),load(prep_path),load(generated_path)
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or cohort['selected_counts']!=dict(train=1024,validation=64,reserved=0):raise ValueError('Invalid fixed cohort')
    entries={r['original_path']:r for r in prep['files']};lookup={name:data/r['prepared_path'] for name,r in entries.items()}
    for name,entry in entries.items():
        if digest(lookup[name])!=entry['sha256']:raise ValueError('Changed prepared input')
    train=cohort['groups']['train'];train_ids={r['index'] for r in train};train_contents={r['content'] for r in train};gen={r['index']:r for r in generated['cases']}
    if not generated['finished'] or generated['cohort_sha256']!=digest(cohort_path) or generated['prepared_manifest_sha256']!=digest(prep_path) or generated['checkpoint_sha256']!=START_SHA or len(generated['cases'])!=1024 or set(gen)!=train_ids:raise ValueError('Changed generation manifest')
    for name,sha in generated['source_sha256'].items():
        if digest(generated_dir/'frozen-source'/name)!=sha:raise ValueError('Changed frozen generation source')
    if digest(Path(__file__).with_name('consistent_reference.py'))!=generated['source_sha256']['consistent_reference.py']:raise ValueError('Changed reference schedule implementation')
    for row in train:
        item=gen[row['index']];parents={row['content'],row['gt'],item['reference_original']}
        if item['content']!=row['content'] or item['gt']!=row['gt'] or item['reference_original'] not in train_contents or item['reference_original']==row['content'] or set(item['parent_sha256'])!=parents or any(entries[p]['sha256']!=sha for p,sha in item['parent_sha256'].items()):raise ValueError('Invalid generation parents')
        if item['generated_reference']!=f'{row["index"]}.png' or digest(generated_dir/item['generated_reference'])!=item['sha256']:raise ValueError('Changed generated reference')
    reused_dirs={'C00':base/'appearance-pairwise-replica-control-v1','C10':base/'appearance-pairwise-replica-v1'};reused={};snapshots={};reused_scores={}
    for name,path in reused_dirs.items():
        record=load(path/'results.json');require_completed_control(record,config,record['source_sha256'])
        if record['appearance_enabled']!=(name=='C10') or record['start_checkpoint_sha256']!=START_SHA or record['cohort_sha256']!=digest(cohort_path) or record['prepared_manifest_sha256']!=digest(prep_path):raise ValueError('Mismatched completed cell')
        for filename,sha in record['source_sha256'].items():
            if digest(path/'frozen-source'/filename)!=sha or digest(Path(__file__).with_name(filename))!=sha:raise ValueError('Changed completed-cell source')
        score_path=path/f'epoch-{record["best_epoch"]}/scores.json';selected_path=path/f'epoch-{record["best_epoch"]}.pth'
        if digest(selected_path)!=record['selected_checkpoint_sha256'] or digest(path/'epoch-0.pth')!=record['initial_checkpoint_sha256']:raise ValueError('Changed completed-cell weights')
        reused[name]=record;reused_scores[name]=load(score_path);snapshots[name]=dict(results_sha256=digest(path/'results.json'),selected_scores_sha256=digest(score_path),selected_checkpoint_sha256=digest(selected_path),initial_checkpoint_sha256=digest(path/'epoch-0.pth'))
    if reused['C00']['source_sha256']!=reused['C10']['source_sha256'] or reused['C00']['baseline']!=reused['C10']['baseline']:raise ValueError('Completed cells do not match')
    out=base/f'appearance-invariance-{cell}-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=sorted(set(['train_appearance_invariance.py','appearance_invariance.py','appearance-invariance-protocol.md','consistent_reference.py',*reused['C00']['source_sha256']]))
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Fixed2x2 appearance/reference augmentation study, consumed real-reference development; C11 sole candidate',cell=cell,artifact_revision='v2',preflight_failure_retained='appearance-invariance-C01-v1 (generated-path error before model initialization/updates)',appearance_enabled=args.appearance,augmentation_enabled=True,config=config,start_checkpoint_sha256=START_SHA,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),generation_manifest_sha256=digest(generated_path),reused_cells=snapshots,source_sha256={p.name:digest(p) for p in frozen.iterdir()},reused_native={},initial_raw_equivalence=[],epochs=[],finished=False,app_promoted=False)
    c01_dir=base/'appearance-invariance-C01-v2'
    if args.appearance:
        c01_path=c01_dir/'results.json';c01=load(c01_path);require_completed_control(c01,config,report['source_sha256'])
        if c01['cell']!='C01' or c01['appearance_enabled'] or not c01['augmentation_enabled'] or any(c01[k]!=report[k] for k in ['generation_manifest_sha256','reused_cells','cohort_sha256','prepared_manifest_sha256','start_checkpoint_sha256']):raise ValueError('Mismatched augmented control')
        if digest(c01_dir/f'epoch-{c01["best_epoch"]}.pth')!=c01['selected_checkpoint_sha256']:raise ValueError('Changed C01 selected weights')
        report['c01_snapshot']=dict(results_sha256=digest(c01_path),selected_checkpoint_sha256=c01['selected_checkpoint_sha256'])
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(path) as im:return torch.from_numpy(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))).permute(2,0,1)
    samples={part:[(r,*[image(lookup[r[role]]) for role in ['content','reference','gt']]) for r in cohort['groups'][part] ] for part in ['train','validation']};synthetic={index:image(generated_dir/item['generated_reference']) for index,item in gen.items()}
    inputs=lambda x:(x.float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    model,api,author=author_model();base_state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in base_state.values()):raise ValueError('Nonfinite starting weights')
    initialize_replication_model(model,base_state);report['author_revision']=author['revision'];initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    old_initial=torch.load(reused_dirs['C00']/'epoch-0.pth',map_location='cpu',weights_only=True)
    if set(initial_state)!=set(old_initial) or any(not torch.equal(v,old_initial[k]) for k,v in initial_state.items()):raise ValueError('Complete initialization differs from C00')
    if args.appearance:
        c01_initial=torch.load(c01_dir/'epoch-0.pth',map_location='cpu',weights_only=True)
        if set(c01_initial)!=set(initial_state) or any(not torch.equal(v,c01_initial[k]) for k,v in initial_state.items()):raise ValueError('Augmented cell initializations differ')
    model.eval()
    with torch.inference_mode():
        for name,path in reused_dirs.items():
            model.load_state_dict(torch.load(path/f'epoch-{reused[name]["best_epoch"]}.pth',map_location='cpu',weights_only=True),strict=True);verified=[]
            for row,c,s,gt in samples['validation']:
                actual=pixels(appearance_forward(model,api,inputs(c[None]),inputs(s[None]),disable_appearance=name=='C00')[0][0]);p=path/f'epoch-{reused[name]["best_epoch"]}'/f'{row["index"]}.png';stored={r['index']:r for r in reused_scores[name]['cases']}
                if digest(p)!=stored[row['index']]['sha256']:raise ValueError('Changed reused PNG')
                with Image.open(p) as im:expected=np.array(im.convert('RGB'))
                maximum=int(np.abs(actual.astype(int)-expected.astype(int)).max())
                if maximum!=0:raise ValueError('Reused native output changed')
                verified.append(dict(index=row['index'],maximum_code_difference=maximum))
            report['reused_native'][name]=verified
    model.load_state_dict(initial_state,strict=True)
    if any(not torch.equal(v.detach().cpu(),initial_state[k]) for k,v in model.state_dict().items()):raise ValueError('Failed to restore untrained initialization')
    report['initial_complete_state_exact_to_C00']=True;report['restored_initial_after_reuse']=True;initial_reference={k:v.detach().cpu().clone() for k,v in model.reference_adapter.state_dict().items()}
    @torch.inference_mode()
    def evaluate(label):
        model.eval();directory=out/label;directory.mkdir();cases=[];groups={}
        for row,c,s,gt in samples['validation']:
            ct,st=inputs(c[None]),inputs(s[None]);raw,grid=appearance_forward(model,api,ct,st,disable_appearance=not args.appearance);prediction=pixels(raw[0]);truth=gt.permute(1,2,0).numpy();p=directory/f'{row["index"]}.png';Image.fromarray(prediction).save(p)
            if label=='baseline-current':
                hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](ct));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](st));original,original_grid=model(ct,hc,hs);finite_output(original,ct.shape);finite_output(original_grid,(1,13,8,16,16));difference=max(float((original-raw).abs().max()),float((original_grid-grid).abs().max()))
                if difference!=0:raise ValueError('Initial raw/grid changed')
                report['initial_raw_equivalence'].append(dict(index=row['index'],maximum_float_difference=difference))
            cases.append(dict(index=row['index'],mae=float(np.abs(prediction.astype(float)-truth).mean()),sha256=digest(p)));group=groups.setdefault(row['group_id'],{})
            if row['style_index'] in group:raise ValueError('Duplicate validation style')
            group[row['style_index']]=(prediction,truth)
        if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()):raise ValueError('Incomplete real-reference evaluation')
        paired=[dict(group_id=name,**response_metrics(g[0][0],g[1][0],g[0][1],g[1][1]),tones=tone_response(g[0][0],g[1][0],g[0][1],g[1][1])) for name,g in groups.items()]
        table=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
        (directory/'scores.json').write_text(json.dumps(table,indent=2),encoding='utf-8');return table
    baseline=evaluate('baseline-current');initial=evaluate('epoch-0');report['baseline']=baseline
    if baseline!=reused['C00']['baseline'] or initial!=baseline:raise ValueError('Initial scores/PNG differ from current13')
    torch.save(model.state_dict(),out/'epoch-0.pth');report['initial_checkpoint_sha256']=digest(out/'epoch-0.pth');(out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(cell=cell,baseline=baseline['mean'],reused_native_cases=128,initial_restored=True)),flush=True)
    extra=list(model.pairwise_adapter.parameters())+list(model.reference_adapter.parameters());ids={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],lr=config['lr'],weight_decay=config['weight_decay'])
    seed();rng=np.random.default_rng(config['seed']);updates=0;best=initial['mean'];best_epoch=0
    for epoch in range(1,9):
        model.train();losses=[];order=[];assignments=[];replaced=0;start=time.perf_counter()
        for indices in grouped_batches(train,rng):
            batch=[samples['train'][i] for i in indices];c=inputs(torch.stack([r[1] for r in batch]));gt=inputs(torch.stack([r[3] for r in batch]));refs=[]
            for row,_,reference,_ in batch:
                use=use_synthetic_reference(row['style_index'],epoch);refs.append(synthetic[row['index']] if use else reference);replaced+=int(use);order.append(row['index']);assignments.append([row['index'],use,gen[row['index']]['sha256'] if use else entries[row['reference']]['sha256']])
            s=inputs(torch.stack(refs));optimizer.zero_grad(set_to_none=True);prediction=appearance_forward(model,api,c,s,disable_appearance=not args.appearance)[0].clamp(0,1)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if not args.appearance and any(p.grad is not None for p in model.reference_adapter.parameters()):raise ValueError('Disabled appearance received gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss))
            if updates%64==0:
                progress=dict(cell=cell,epoch=epoch,updates=updates,loss=float(loss));(out/'progress.json').write_text(json.dumps(progress),encoding='utf-8');print(json.dumps(progress),flush=True)
        order_sha=hashlib.sha256(json.dumps(order).encode()).hexdigest();assignment_sha=hashlib.sha256(json.dumps(assignments).encode()).hexdigest()
        if replaced!=512 or len(order)!=1024 or set(order)!=train_ids or order_sha!=reused['C00']['epochs'][epoch-1]['target_order_sha256']:raise ValueError('Target stream or reference schedule differs')
        if args.appearance and (assignment_sha!=c01['epochs'][epoch-1]['reference_assignment_sha256'] or order_sha!=c01['epochs'][epoch-1]['target_order_sha256']):raise ValueError('C01/C11 reference inputs differ')
        if not args.appearance and any(not torch.equal(v.detach().cpu(),initial_reference[k]) for k,v in model.reference_adapter.state_dict().items()):raise ValueError('Unused appearance changed')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite trained state')
        p=out/f'epoch-{epoch}.pth';torch.save(model.state_dict(),p);validation=evaluate(f'epoch-{epoch}')
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        row=dict(epoch=epoch,updates=updates,train_loss=float(np.mean(losses)),training_seconds=seconds,validation_mean=validation['mean'],delta_mean=validation['delta_mean'],checkpoint_sha256=digest(p),target_order_sha256=order_sha,reference_assignment_sha256=assignment_sha,synthetic_reference_count=replaced)
        if not args.appearance:row['unused_appearance_unchanged']=True
        report['epochs'].append(row);report.update(best_epoch=best_epoch,best_validation_mean=best);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    if updates!=2048 or digest(checkpoint)!=START_SHA or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(generated_path)!=report['generation_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Budget or frozen evidence changed')
    for name,path in reused_dirs.items():
        if digest(path/'results.json')!=snapshots[name]['results_sha256'] or digest(path/f'epoch-{reused[name]["best_epoch"]}/scores.json')!=snapshots[name]['selected_scores_sha256']:raise ValueError('Reused cell changed')
    if args.appearance and digest(c01_path)!=report['c01_snapshot']['results_sha256']:raise ValueError('C01 changed')
    selected=load(out/f'epoch-{best_epoch}/scores.json');gate=invariance_gate(baseline,selected,reused_scores['C00'],reused_scores['C10']);report['development_gate']=dict(gate,measured_gate_passed=gate['proceed_to_independent_evaluation'],candidate_cell=args.appearance,proceed_to_independent_evaluation=bool(args.appearance and gate['proceed_to_independent_evaluation']));report['selected_checkpoint_sha256']=digest(out/f'epoch-{best_epoch}.pth');report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['development_gate']),flush=True)

if __name__=='__main__':main()
