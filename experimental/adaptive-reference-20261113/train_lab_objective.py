"""Matched primary-image objectives; no extra head or auxiliary teacher."""
from functools import lru_cache
import hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from lab_objective import color_objective
from coverage_training import block_batches
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';load=lambda p:json.loads(p.read_text())
    probe=base/'lab-objective-probe-v1/results.json';prep=load(probe)
    assert prep['finished'] and prep['model_state_unchanged'] and not prep['training']
    for n,h in prep['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=base/'cohort-training-coverage-v2';assert digest(data/'cohort.json')==prep['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==prep['manifest_sha256']
    cohort=load(data/'cohort.json');lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']};train=cohort['groups']['train'];dev=cohort['groups']['validation']
    assert len(train)==4096 and len(dev)==64
    warm=base/'training-coverage-expanded-v2';warm_record=load(warm/'results.json');checkpoint=warm/'epoch-4.pth';assert digest(checkpoint)==prep['checkpoint_sha256']
    for n,h in warm_record['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    start_scores=load(warm/'epoch-4/scores.json');prior=base/'appearance-pairwise-replica-v1';original_scores=load(prior/'epoch-7/scores.json');baseline=warm_record['baseline']
    out=base/'lab-objective-training-v1';out.mkdir(exist_ok=False);(out/'frozen-source').mkdir()
    names=set(warm_record['source_sha256'])|{'train_lab_objective.py','lab_objective.py','lab-objective-training-protocol.md','test_lab_objective.py','probe_lab_objective.py','lab-objective-probe-protocol.md'}
    for n in sorted(names):(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    config=dict(seed=20261202,lr=1e-5,adapter_lr=1e-4,weight_decay=1e-4,batch=4,epochs=4,max_updates=4096,gradient_weight=.1,color_scale=prep['summary']['frozen_color_scale'],gradient_clip=1.)
    protected=[data/'cohort.json',data/'srgb-v1/manifest.json',probe,checkpoint,warm/'results.json',warm/'epoch-4/scores.json',prior/'epoch-7/scores.json']
    report=dict(scope='Consumed train4096/development64 RGB MSE versus training-RMS-scaled D50 Lab main loss; no independent acceptance',config=config,
        source_sha256={n:digest(out/'frozen-source'/n) for n in names},protected_sha256={str(p.relative_to(base)):digest(p) for p in protected},
        arms={},training=True,finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    @lru_cache(maxsize=768)
    def image(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert p.resolve().is_relative_to(base.resolve()) and digest(p)==e['sha256']
        with Image.open(p) as im:a=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return torch.from_numpy(a).permute(2,0,1)
    tensor=lambda a:(a.float()/255).cuda()
    truths={r['index']:image(r['gt']).permute(1,2,0).numpy() for r in dev}
    def seed():random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    shared_orders=None
    for arm in ['rgb_control','lab_candidate']:
        seed();model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True))
        initial=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(initial,strict=True)
        assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
        folder=out/arm;folder.mkdir();torch.save(model.state_dict(),folder/'epoch-0.pth')
        record=dict(epochs=[],finished=False,initial_model_state_exact=True,best_epoch=0,best_mean=None)
        @torch.inference_mode()
        def evaluate(epoch):
            model.eval();directory=folder/f'epoch-{epoch}';directory.mkdir();predictions={}
            for row in dev:
                c,s=[tensor(image(row[k])[None]) for k in ['content','reference']]
                raw,_=appearance_forward(model,api,c,s);pixels=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                if epoch==0:
                    old=warm/'epoch-4'/f"{row['index']}.png";expected=next(r for r in start_scores['cases'] if r['index']==row['index']);assert digest(old)==expected['sha256']
                    with Image.open(old) as im:assert np.array_equal(pixels,np.array(im))
                predictions[row['index']]=pixels;Image.fromarray(pixels).save(directory/f"{row['index']}.png")
            scores=score_outputs(dev,predictions,truths)
            for row in scores['cases']:row['sha256']=digest(directory/f"{row['index']}.png")
            (directory/'scores.json').write_text(json.dumps(scores,indent=2));return scores
        first=evaluate(0)
        def without_hashes(s):return dict(s,cases=[{k:v for k,v in r.items() if k!='sha256'} for r in s['cases']])
        assert without_hashes(first)==without_hashes(start_scores)
        record['best_mean']=first['mean'];record['initial_full_scores_exact']=True;report['arms'][arm]=record
        (out/'progress.json').write_text(json.dumps(report,indent=2))
        extra=list(model.reference_adapter.parameters())+list(model.pairwise_adapter.parameters());extra_ids={id(p) for p in extra}
        optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in extra_ids],lr=config['lr']),dict(params=extra,lr=config['adapter_lr'])],weight_decay=config['weight_decay'])
        seed();rng=np.random.default_rng(config['seed']);orders=[];updates=0
        for epoch in range(1,5):
            model.train();batches=block_batches(train,rng,1);assert len(batches)==1024
            values=[];order=[];start=time.perf_counter()
            for indices in batches:
                batch=[train[i] for i in indices];order.extend(r['index'] for r in batch)
                c,s,gt=[tensor(torch.stack([image(r[k]) for r in batch])) for k in ['content','reference','gt']]
                optimizer.zero_grad(set_to_none=True);pred=appearance_forward(model,api,c,s)[0].clamp(0,1)
                rgb=F.mse_loss(pred,gt);edge=F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1])
                main_loss=rgb if arm=='rgb_control' else config['color_scale']*color_objective(pred,gt)
                loss=main_loss+config['gradient_weight']*edge;assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()),1,error_if_nonfinite=True);optimizer.step();updates+=1
                values.append([float(loss.detach()),float(rgb.detach()),float(edge.detach()),float(main_loss.detach())])
                if updates%256==0:print(json.dumps(dict(arm=arm,epoch=epoch,updates=updates,loss=values[-1][0])),flush=True)
            assert len(order)==len(set(order))==4096
            order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest();orders.append(order_hash)
            torch.save(model.state_dict(),folder/f'epoch-{epoch}.pth');scores=evaluate(epoch)
            if scores['mean']<record['best_mean']:record.update(best_epoch=epoch,best_mean=scores['mean'])
            record['epochs'].append(dict(epoch=epoch,updates=updates,loss_means=np.mean(values,axis=0).tolist(),order_sha256=order_hash,mean=scores['mean'],delta_mean=scores['delta_mean'],seconds=time.perf_counter()-start,checkpoint_sha256=digest(folder/f'epoch-{epoch}.pth')))
            (out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(arm=arm,epoch=epoch,mean=scores['mean'],selected=record['best_epoch'])),flush=True)
        if shared_orders is None:shared_orders=orders
        else:assert orders==shared_orders
        best_scores=load(folder/f"epoch-{record['best_epoch']}/scores.json")
        comparisons={'original_C10':original_scores,'expanded_start':start_scores}
        if arm=='lab_candidate':
            control=report['arms']['rgb_control'];assert control['finished'];comparisons['matched_control']=load(out/'rgb_control'/f"epoch-{control['best_epoch']}/scores.json")
        gates={name:response_development_gate(baseline,best_scores,comparison) for name,comparison in comparisons.items()}
        record.update(finished=True,gates=gates,selected_checkpoint_sha256=digest(folder/f"epoch-{record['best_epoch']}.pth"),full_candidate_gate=bool(arm=='lab_candidate' and all(g['proceed_to_independent_evaluation'] for g in gates.values())))
        (out/'progress.json').write_text(json.dumps(report,indent=2))
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    for n,h in report['protected_sha256'].items():assert digest(base/n)==h
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(finished=True,arms={k:dict(best_epoch=v['best_epoch'],mean=v['best_mean'],full_candidate_gate=v['full_candidate_gate']) for k,v in report['arms'].items()})),flush=True)


if __name__=='__main__':main()
