"""Matched head-only training with cached frozen descriptors; development only."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import freeze_and_create,condition_forward
from check_petal_condition import state_hash
from paired_training import grouped_batches
from multiref_response import response_metrics


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=dict(seed=20261125,lr=1e-4,weight_decay=1e-4,epochs=8,batch=4,max_updates=2048,gradient_weight=.1,gradient_clip=1.)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';cache=base/'petal-condition-features-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');features_report=load(cache/'results.json');bridge_path=base/'petal-condition-bridge-v1/results.json';bridge=load(bridge_path)
    if not bridge['finished'] or len(bridge['cases'])!=64 or any(r['maximum_zero_float_difference']!=0 or r['maximum_zero_grid_difference']!=0 or not r['disabled_exact'] for r in bridge['cases']):raise ValueError('Missing exact bridge')
    if digest(cache/'features.npz')!=features_report['features_sha256'] or digest(cache/'results.json')!=bridge['feature_report_sha256'] or digest(data/'cohort.json')!=features_report['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=features_report['prepared_manifest_sha256']:raise ValueError('Data linkage changed')
    for n,sha in bridge['source_sha256'].items():
        if digest(Path(__file__).with_name(n))!=sha:raise ValueError('Bridge source changed')
    with np.load(cache/'features.npz') as z:petal=z['PETAL1024'].copy()
    lookup={v['original_path']:v for v in prep['files']};pixels={}
    for split in ['train','validation']:
        for row in cohort['groups'][split]:
            for role in ['content','reference','gt']:
                name=row[role]
                if name in pixels:continue
                entry=lookup[name];path=data/entry['prepared_path']
                if digest(path)!=entry['sha256']:raise ValueError('Training pixel changed')
                with Image.open(path) as im:pixels[name]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    out=base/'petal-condition-training-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['train_petal_condition.py','petal_condition.py','check_petal_condition.py','petal-condition-protocol.md','petal-condition-training-addendum.md','paired_training.py','multiref_response.py','train_triplet_pilot.py','training_checks.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=bridge['checkpoint_sha256']:raise ValueError('Wrong baseline')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    if before!=bridge['backbone_state_after_sha256']:raise ValueError('Wrong frozen backbone')
    report=dict(scope='Consumed1024train/64development head-only comparison, not independent acceptance',config=config,source_sha256={n:digest(frozen/n) for n in names},checkpoint_sha256=digest(cp),bridge_results_sha256=digest(bridge_path),feature_report_sha256=digest(cache/'results.json'),petal_features_sha256=digest(cache/'features.npz'),backbone_state_before=before,arms={},finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def tensor(names):return torch.from_numpy(np.stack([pixels[n] for n in names])).permute(0,3,1,2).float().cuda()/255
    control=np.zeros_like(petal)
    with torch.inference_mode():
        unique=features_report['unique_inputs']
        for offset in range(0,len(unique),4):
            batch=unique[offset:offset+4];rgb=[]
            for item in batch:
                with Image.open(ROOT/item['path']) as im:rgb.append(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
            value=torch.from_numpy(np.stack(rgb)).permute(0,3,1,2).float().cuda()/255
            h=F.normalize(model.img_projection(api['TF'].resize(value,(64,64),antialias=True)).mean((-2,-1)),dim=1,eps=1e-8).cpu().numpy()
            for item,v in zip(batch,h):control[item['index'],:128]=v
    if not np.isfinite(control).all() or not np.allclose(np.linalg.norm(control,axis=1),1,atol=1e-5):raise ValueError('Invalid H128 controls')
    np.savez_compressed(out/'control-features.npz',H128_PAD1024=control);report['control_features_sha256']=digest(out/'control-features.npz')
    records={s:[r for r in features_report['rows'] if r['split']==s] for s in ['train','validation']}
    for split in records:
        if len(records[split])!=len(cohort['groups'][split]):raise ValueError('Role mapping length changed')
        for row,r in zip(cohort['groups'][split],records[split]):
            if (row['index'],row['group_id'],row['style_index'])!=(r['index'],r['group_id'],r['style_index']) or any(row[k]!=r['roles'][k]['original_path'] for k in ['content','reference']):raise ValueError('Feature row changed')
    for arm,features in [('H128',control),('PETAL1024',petal)]:
        torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);head=freeze_and_create(model);folder=out/arm;folder.mkdir();rng=np.random.default_rng(config['seed']);arm_report=dict(epochs=[],best_epoch=0,best_mean=None,finished=False)
        def forward(indices,split):
            rows=[cohort['groups'][split][i] for i in indices];rr=[records[split][i] for i in indices]
            c,s=[tensor([r[k] for r in rows]) for k in ['content','reference']]
            zc,zr=[torch.from_numpy(np.stack([features[r['roles'][k]['feature_index']] for r in rr])).cuda() for k in ['content','reference']]
            return condition_forward(model,head,api,c,s,zc,zr)[0],rows
        @torch.inference_mode()
        def evaluate(epoch):
            head.eval();directory=folder/f'epoch-{epoch}';directory.mkdir();cases=[];groups={}
            for i in range(64):
                raw,rows=forward([i],'validation');row=rows[0];pred=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8);truth=pixels[row['gt']];path=directory/f'{row["index"]}.png';Image.fromarray(pred).save(path)
                cases.append(dict(index=row['index'],style_index=row['style_index'],mae=float(np.abs(pred.astype(float)-truth.astype(float)).mean()),sha256=digest(path)))
                groups.setdefault(row['group_id'],{})[row['style_index']]=(pred,truth)
            paired=[dict(group_id=k,**response_metrics(v[0][0],v[1][0],v[0][1],v[1][1])) for k,v in groups.items()]
            scores=dict(mean=float(np.mean([r['mae'] for r in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([r['delta_mae'] for r in paired])))
            (directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');return scores
        initial=evaluate(0);arm_report.update(baseline=initial,best_mean=initial['mean']);torch.save(head.state_dict(),folder/'epoch-0.pth');optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        for epoch in range(1,9):
            head.train();losses=[];start=time.perf_counter();batches=grouped_batches(cohort['groups']['train'],rng)
            if len(batches)!=256:raise ValueError('Wrong update count')
            for indices in batches:
                optimizer.zero_grad(set_to_none=True);raw,rows=forward(indices,'train');pred=raw.clamp(0,1);gt=tensor([r['gt'] for r in rows])
                loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
                if not torch.isfinite(loss):raise ValueError('Invalid training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
            scores=evaluate(epoch);path=folder/f'epoch-{epoch}.pth';torch.save(head.state_dict(),path)
            if scores['mean']<arm_report['best_mean']:arm_report.update(best_epoch=epoch,best_mean=scores['mean'])
            row=dict(epoch=epoch,updates=epoch*256,loss=float(np.mean(losses)),validation_mean=scores['mean'],delta_mean=scores['delta_mean'],seconds_including_evaluation=time.perf_counter()-start,head_sha256=digest(path));arm_report['epochs'].append(row);report['arms'][arm]=arm_report
            (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(arm=arm,**row)),flush=True)
        arm_report['finished']=True
        if state_hash(model)!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Backbone changed')
        report['arms'][arm]=arm_report
    def scores(arm):return load(out/arm/f"epoch-{report['arms'][arm]['best_epoch']}/scores.json")
    candidate,matched=scores('PETAL1024'),scores('H128');baseline=report['arms']['PETAL1024']['baseline']
    def compare(a,b,ratio,count):
        aa={r['index']:r['mae'] for r in a['cases']};bb={r['index']:r['mae'] for r in b['cases']}
        return dict(mean_gain=1-b['mean']/a['mean'],improved=sum(bb[k]<v-.1 for k,v in aa.items()),severe=sum(bb[k]>v+5 and bb[k]>v*1.25 for k,v in aa.items()),passed=b['mean']<=a['mean']*(1-ratio) and sum(bb[k]<v-.1 for k,v in aa.items())>=count and sum(bb[k]>v+5 and bb[k]>v*1.25 for k,v in aa.items())<=2)
    gates=dict(vs_current=compare(baseline,candidate,.05,40),vs_matched=compare(matched,candidate,.03,36))
    added_a=[r for r in baseline['cases'] if r['style_index']==1];added_b={r['index']:r['mae'] for r in candidate['cases'] if r['style_index']==1}
    gates['added_reference_passed']=bool(np.mean(list(added_b.values()))<=np.mean([r['mae'] for r in added_a]) and sum(added_b[r['index']]<r['mae']-.1 for r in added_a)>=17)
    before_delta={r['group_id']:r['delta_mae'] for r in matched['paired']};after_delta={r['group_id']:r['delta_mae'] for r in candidate['paired']}
    gates['response_vs_matched_passed']=candidate['delta_mean']<=.95*matched['delta_mean'] and sum(after_delta[k]<v-.1 for k,v in before_delta.items())>=20
    gates['passed']=gates['vs_current']['passed'] and gates['vs_matched']['passed'] and bool(gates['added_reference_passed']) and gates['response_vs_matched_passed']
    report['gates']=gates;report['backbone_state_after']=state_hash(model)
    if any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()) or digest(bridge_path)!=report['bridge_results_sha256']:raise ValueError('Frozen evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
