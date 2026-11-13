"""Bounded academic fine-tuning; validation is development, never acceptance."""
import ast
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from training_checks import finite_output
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping


def author_model():
    source=ROOT/'.tools/Hist2Style-source';manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision']!='a3ef9ba0455d957b54c459c88cac36cf5740c164':raise ValueError('Author revision changed')
    for entry in manifest['files']:
        if digest(source/entry['path'])!=entry['sha256']:raise ValueError('Author asset changed')
    tree=ast.parse((source/'demo/model.py').read_text(encoding='utf-8'));body=[];omitted=[]
    for node in tree.body:
        if isinstance(node,ast.Import) and any(x.name=='lightning' for x in node.names):omitted.append('lightning')
        elif isinstance(node,ast.ClassDef) and node.name=='Model':omitted.append('Model')
        else:body.append(node)
    if omitted!=['lightning','Model']:raise ValueError('Unexpected author wrapper')
    tree.body=body;api={'__name__':'research_hist2style'};exec(compile(tree,str(source/'demo/model.py'),'exec'),api)
    model=api['Grid']();payload=torch.load(source/'demo/model-epoch=1500-val_loss=2.79.ckpt',map_location='cpu',weights_only=True)
    if any(not k.startswith('model.') for k in payload['state_dict']):raise ValueError('Unexpected checkpoint namespace')
    state={k[6:]:v for k,v in payload['state_dict'].items()};expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Incomplete weights')
    for k,v in state.items():
        if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all():raise ValueError('Invalid weight')
    model.load_state_dict(state,strict=True)
    return model.to('cuda'),api,manifest


def main():
    config=dict(seed=20261113,lr=3e-5,weight_decay=1e-4,batch=4,epochs=8,size=256,gradient_weight=.1,gradient_clip=1.,max_updates=256)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=ROOT/'outputs/adaptive-reference-20261113/tst100k/cohort-v1';cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'))
    prepared=data/'srgb-v1/manifest.json';prep=json.loads(prepared.read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(data/'cohort.json'):raise ValueError('Data not ready')
    lookup={e['original_path']:data/e['prepared_path'] for e in prep['files']}
    for e in prep['files']:
        if digest(lookup[e['original_path']])!=e['sha256']:raise ValueError('Prepared image changed')
    model,api,author=author_model()
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/training-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['train_triplet_pilot.py','triplet_color.py','triplet-training-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Noncommercial academic pilot; validation is development; no reserved inputs or product integration',config=config,
        author=author,cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(prepared),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},epochs=[])
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def image(path):
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    samples={}
    for part in ['train','validation']:
        samples[part]=[]
        for row in cohort['groups'][part]:
            tensors=[torch.from_numpy(image(lookup[row[role]]).copy()).permute(2,0,1).float()/255 for role in ['content','reference','gt']]
            samples[part].append((row,*tensors))
    def forward(c,s):
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        return finite_output(model(c,hc,hs)[0],c.shape).clamp(0,1)
    def rgb(x):return np.rint(finite_output(x,(3,256,256)).detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    @torch.inference_mode()
    def evaluate(name):
        model.eval();directory=out/name;directory.mkdir();rows=[]
        for row,c,s,gt in samples['validation']:
            start=time.perf_counter();pred=forward(c[None].cuda(),s[None].cuda())[0];torch.cuda.synchronize();seconds=time.perf_counter()-start
            pixels=rgb(pred);path=directory/f'{row["index"]}.png';Image.fromarray(pixels).save(path)
            rows.append(dict(index=row['index'],mae=float(np.abs(pixels.astype(float)-rgb(gt)).mean()),seconds=seconds,sha256=digest(path)))
        result=dict(mean=float(np.mean([r['mae'] for r in rows])),cases=rows)
        (directory/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
    before=evaluate('before');current=[]
    base_dir=out/'current';base_dir.mkdir()
    for row,c,s,gt in samples['validation']:
        mapping=analyze(lookup[row['content']],lookup[row['reference']]);pixels=apply_mapping(rgb(c),mapping,1)
        path=base_dir/f'{row["index"]}.png';Image.fromarray(pixels).save(path)
        current.append(dict(index=row['index'],mae=float(np.abs(pixels.astype(float)-rgb(gt)).mean()),sha256=digest(path)))
    report.update(before=before,current=dict(mean=float(np.mean([r['mae'] for r in current])),cases=current))
    (out/'baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(before=before['mean'],current=report['current']['mean'])),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);order_rng=np.random.default_rng(config['seed']);steps=0;best=before['mean'];best_epoch=0
    torch.save(model.state_dict(),out/'epoch-0.pth')
    for epoch in range(1,9):
        model.train();losses=[];start=time.perf_counter()
        order=order_rng.permutation(len(samples['train']))
        for offset in range(0,len(order),4):
            selected=[samples['train'][int(j)] for j in order[offset:offset+4]]
            c,s,gt=[torch.stack([r[k] for r in selected]).cuda() for k in [1,2,3]]
            optimizer.zero_grad(set_to_none=True);prediction=forward(c,s)
            loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss));steps+=1
        torch.cuda.synchronize();training_seconds=time.perf_counter()-start
        torch.save(model.state_dict(),out/f'epoch-{epoch}.pth');validation=evaluate(f'epoch-{epoch}')
        if validation['mean']<best:best,best_epoch=validation['mean'],epoch
        record=dict(epoch=epoch,updates=steps,train_loss=float(np.mean(losses)),training_seconds=training_seconds,validation_mean=validation['mean'],checkpoint_sha256=digest(out/f'epoch-{epoch}.pth'))
        report['epochs'].append(record);report.update(best_epoch=best_epoch,best_validation_mean=best,finished=epoch==8)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    if steps!=256:raise ValueError('Unexpected training updates')
    print(json.dumps(dict(finished=True,best_epoch=best_epoch,best_validation_mean=best)),flush=True)


if __name__=='__main__':main()
