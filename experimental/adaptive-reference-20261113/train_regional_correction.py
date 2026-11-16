"""Fixed matched residual-head training; no app integration or independent acceptance."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from regional_correction import RegionalCorrection,global_control
from paired_training import grouped_batches
from multiref_response import response_metrics


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';cache=base/'regional-correction-inputs-v1'
    report_path=cache/'results.json';registry=json.loads(report_path.read_text());bridge_path=base/'regional-correction-bridge-v1/results.json'
    bridge=json.loads(bridge_path.read_text());assert bridge['finished'] and bridge['input_report_sha256']==digest(report_path)
    for n,h in bridge['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    assert digest(cache/'descriptors.npz')==registry['descriptors_sha256']
    with np.load(cache/'descriptors.npz') as a:features=a['regional'].copy()
    inputs={x['index']:x for x in registry['inputs']};rows={s:[x for x in registry['rows'] if x['split']==s] for s in ['train','validation']}
    data=base/'cohort-multiref-v1';cohort=json.loads((data/'cohort.json').read_text());assert digest(data/'cohort.json')==registry['cohort_sha256']
    manifest=data/'srgb-v1/manifest.json';assert digest(manifest)==registry['prepared_manifest_sha256'];lookup={x['original_path']:x for x in json.loads(manifest.read_text())['files']}
    out=base/'regional-correction-training-v1';out.mkdir(exist_ok=False);(out/'frozen-source').mkdir()
    names=['train_regional_correction.py','regional_correction.py','regional-correction-training-protocol.md','paired_training.py','multiref_response.py','spatial_render.py']
    for n in names:(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    config=dict(seed=20261127,epochs=8,batch=4,lr=1e-4,weight_decay=1e-4,gradient_weight=.1,gradient_clip=1.)
    report=dict(scope='Consumed train1024/development64 matched global+regional affine correction',config=config,source_sha256={n:digest(out/'frozen-source'/n) for n in names},input_report_sha256=digest(report_path),bridge_sha256=digest(bridge_path),arms={},finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    engine=SpatialEngine('multiref-v1');assert engine.checkpoint_sha==bridge['model_sha256']
    samples={};raw_hashes=[]
    def read(p,sha):
        assert digest(p)==sha
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def truth(row):
        e=lookup[row['gt']];return read(data/e['prepared_path'],e['sha256'])
    for split in ['train','validation']:
        samples[split]=[]
        for i,row in enumerate(rows[split]):
            source,ref=[inputs[row['roles'][role]['feature_index']] for role in ['content','reference']]
            x,y=[read(ROOT/e['path'],e['parent_sha256']) for e in [source,ref]]
            assert digest(cache/source['mask_path'])==source['mask_sha256'];p=np.load(cache/source['mask_path'])
            prepared=engine.prepare(x,y);raw=prepared.float_output(x)
            assert np.array_equal(np.rint(raw.clip(0,1)*255).astype(np.uint8),prepared.byte_output(x))
            sample=dict(rgb=x,raw=raw,probability=p,content=features[source['index']],reference=features[ref['index']])
            if split=='train':sample['truth']=truth(cohort['groups'][split][i])
            samples[split].append(sample)
            raw_hashes.append(__import__('hashlib').sha256(raw.tobytes()).hexdigest())
            if (i+1)%128==0:print(json.dumps(dict(preparing=split,count=i+1)),flush=True)
    report['raw_cache_hashes']=raw_hashes
    tensor=lambda x:torch.from_numpy(np.stack(x)).to(engine.device)
    def image_tensor(x):return tensor(x).permute(0,3,1,2).float()/255
    for arm in ['control','regional']:
        torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);head=RegionalCorrection().to(engine.device)
        folder=out/arm;folder.mkdir();rng=np.random.default_rng(config['seed']);record=dict(epochs=[],best_epoch=0,best_mean=None,finished=False)
        def forward(indices,split):
            batch=[samples[split][i] for i in indices];x=image_tensor([s['rgb'] for s in batch]);raw=tensor([s['raw'] for s in batch]).permute(0,3,1,2)
            p=tensor([s['probability'] for s in batch])[:,None];c=tensor([s['content'] for s in batch]);r=tensor([s['reference'] for s in batch])
            if arm=='control':c,r=global_control(c),global_control(r)
            return head(x,raw,p,c,r)
        @torch.inference_mode()
        def evaluate(epoch):
            head.eval();directory=folder/f'epoch-{epoch}';directory.mkdir();predictions=[]
            for i,row in enumerate(rows['validation']):
                actual=forward([i],'validation');rgb=np.rint(actual[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                p=directory/f"{row['index']}.png";Image.fromarray(rgb).save(p);predictions.append((rgb,p))
            cases=[];groups={}
            for i,(rgb,p) in enumerate(predictions):
                row=cohort['groups']['validation'][i];gt=truth(row)
                cases.append(dict(index=row['index'],style_index=row['style_index'],mae=float(np.abs(rgb.astype(float)-gt).mean()),sha256=digest(p)))
                groups.setdefault(row['group_id'],{})[row['style_index']]=(rgb,gt)
            paired=[dict(group_id=k,**response_metrics(v[0][0],v[1][0],v[0][1],v[1][1])) for k,v in groups.items()]
            result=dict(mean=float(np.mean([x['mae'] for x in cases])),cases=cases,paired=paired,delta_mean=float(np.mean([x['delta_mae'] for x in paired])))
            (directory/'scores.json').write_text(json.dumps(result,indent=2));return result
        initial=evaluate(0);record['best_mean']=initial['mean'];record['baseline']=initial
        torch.save(head.state_dict(),folder/'epoch-0.pth');optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        for epoch in range(1,9):
            head.train();losses=[];start=time.perf_counter();batches=grouped_batches(cohort['groups']['train'],rng);assert len(batches)==256
            for indices in batches:
                optimizer.zero_grad(set_to_none=True);pred=forward(indices,'train').clamp(0,1);gt=image_tensor([samples['train'][i]['truth'] for i in indices])
                loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            scores=evaluate(epoch);p=folder/f'epoch-{epoch}.pth';torch.save(head.state_dict(),p)
            if scores['mean']<record['best_mean']:record.update(best_mean=scores['mean'],best_epoch=epoch)
            record['epochs'].append(dict(epoch=epoch,updates=epoch*256,loss=float(np.mean(losses)),mean=scores['mean'],delta_mean=scores['delta_mean'],checkpoint_sha256=digest(p),seconds=time.perf_counter()-start))
            report['arms'][arm]=record;(out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(arm=arm,epoch=epoch,mean=scores['mean'],delta=scores['delta_mean'])),flush=True)
        record['finished']=True;report['arms'][arm]=record
    assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(finished=True,best={k:v['best_epoch'] for k,v in report['arms'].items()})),flush=True)


if __name__=='__main__':main()
