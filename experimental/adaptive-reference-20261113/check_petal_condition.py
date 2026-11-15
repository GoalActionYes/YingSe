"""All64 development zero/disabled parity plus live head-only gradient bridge."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import freeze_and_create,condition_forward


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.manual_seed(20261125);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';cache_dir=base/'petal-condition-features-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    evidence=load(cache_dir/'results.json');cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json')
    if not evidence['finished'] or digest(cache_dir/'features.npz')!=evidence['features_sha256'] or digest(data/'cohort.json')!=evidence['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=evidence['prepared_manifest_sha256']:raise ValueError('Feature data changed')
    with np.load(cache_dir/'features.npz') as z:features=z['PETAL1024'].copy()
    lookup={v['original_path']:v for v in prep['files']};records=[r for r in evidence['rows'] if r['split']=='validation']
    if len(records)!=64:raise ValueError('Incomplete development set')
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Wrong baseline')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);head=freeze_and_create(model);before=state_hash(model)
    out=base/'petal-condition-bridge-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['check_petal_condition.py','petal_condition.py','test_petal_condition.py','petal-condition-protocol.md','train_triplet_pilot.py','training_checks.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    result=dict(scope='Consumed64 development exact bridge only, no quality improvement or training',source_sha256={n:digest(frozen/n) for n in names},checkpoint_sha256=digest(cp),features_sha256=evidence['features_sha256'],feature_report_sha256=digest(cache_dir/'results.json'),cases=[],training_started=False,finished=False)
    first=None
    for row,record in zip(cohort['groups']['validation'],records):
        tensors=[];zs=[]
        if (row['index'],row['group_id'],row['style_index'])!=(record['index'],record['group_id'],record['style_index']):raise ValueError('Feature order differs')
        for role in ['content','reference']:
            entry=lookup[row[role]];path=data/entry['prepared_path'];rr=record['roles'][role]
            if row[role]!=rr['original_path'] or digest(path)!=entry['sha256'] or entry['sha256']!=rr['parent_sha256']:raise ValueError('Bridge input changed')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=rr['pixel_sha256']:raise ValueError('Pixel mismatch')
            tensors.append(torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255);zs.append(torch.from_numpy(features[rr['feature_index']])[None].cuda())
        c,s=tensors;zc,zr=zs
        with torch.inference_mode():
            a,ag=condition_forward(model,head,api,c,s,enabled=False);b,bg=condition_forward(model,head,api,c,s,zc,zr)
            if not torch.equal(a,b) or not torch.equal(ag,bg):raise ValueError('Zero head changed baseline')
            head.layers[-1].bias.fill_(.01)
            changed,_=condition_forward(model,head,api,c,s,zc,zr);disabled,_=condition_forward(model,head,api,c,s,enabled=False)
            if not torch.equal(a,disabled):raise ValueError('Disabled head changed baseline')
            delta=float((changed-a).abs().max());head.layers[-1].bias.zero_()
        result['cases'].append(dict(index=row['index'],maximum_zero_float_difference=0.,maximum_zero_grid_difference=0.,disabled_exact=True,nonzero_head_maximum_difference=delta))
        if first is None:first=(c,s,zc,zr)
    c,s,zc,zr=first;head.zero_grad(set_to_none=True);prediction,_=condition_forward(model,head,api,c,s,zc,zr);prediction.mean().backward()
    gradient=float(head.layers[-1].weight.grad.abs().sum())
    if not gradient>0 or any(p.grad is not None or p.requires_grad for p in model.parameters()) or state_hash(model)!=before:raise ValueError('Frozen backbone or head gradient contract failed')
    if not any(r['nonzero_head_maximum_difference']>0 for r in result['cases']):raise ValueError('Head has no effect')
    if any(digest(Path(__file__).with_name(n))!=sha for n,sha in result['source_sha256'].items()):raise ValueError('Frozen source changed')
    result.update(head_weight_gradient_l1=gradient,backbone_state_before_sha256=before,backbone_state_after_sha256=state_hash(model),all_backbone_gradients_none=True,finished=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,cases=64,zero_and_disabled_exact=True,head_gradient_l1=gradient,training_started=False)),flush=True)


if __name__=='__main__':main()
