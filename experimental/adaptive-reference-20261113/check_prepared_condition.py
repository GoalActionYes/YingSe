"""All64 consumed development forward/gradient cache bridge; no optimizer."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import condition_forward
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']}
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Wrong baseline')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    out=base/'prepared-condition-bridge-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['check_prepared_condition.py','prepared_condition.py','petal_condition.py','train_triplet_pilot.py','training_checks.py','check_petal_condition.py','prepared-condition-protocol.md']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Fixed-input forward/gradient cache bridge only; no quality acceptance',source_sha256={n:digest(frozen/n) for n in names},checkpoint_sha256=digest(cp),cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),cases=[],finished=False,optimizer_updates=0)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for offset in range(0,64,4):
        rows=cohort['groups']['validation'][offset:offset+4];arrays=[]
        for role in ['content','reference']:
            batch=[]
            for row in rows:
                entry=lookup[row[role]];path=data/entry['prepared_path']
                if digest(path)!=entry['sha256']:raise ValueError('Pixels changed')
                with Image.open(path) as im:batch.append(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
            arrays.append(torch.from_numpy(np.stack(batch)).permute(0,3,1,2).float().cuda()/255)
        c,s=arrays;torch.cuda.synchronize();start=time.perf_counter();prepared=PreparedCondition(model,api,c,s);torch.cuda.synchronize();prep_seconds=time.perf_counter()-start
        timings={k:[] for k in ['original','cached']};differences=[]
        for level in [0.,.2,-.35]:
            d=(torch.linspace(-1,1,128,device='cuda')[None].repeat(4,1)*level).requires_grad_();a,ag=condition_forward(model,lambda _a,_b:d,api,c,s);ga=torch.autograd.grad(a.square().mean(),d)[0]
            e=d.detach().clone().requires_grad_();b,bg=prepared(e);gb=torch.autograd.grad(b.square().mean(),e)[0]
            d2=d.detach().clone().requires_grad_();again,_=condition_forward(model,lambda _a,_b:d2,api,c,s);gagain=torch.autograd.grad(again.square().mean(),d2)[0]
            values=dict(output=float((a-b).abs().max()),grid=float((ag-bg).abs().max()),gradient=float((ga-gb).abs().max()),original_repeat_gradient_difference=float((ga-gagain).abs().max()),original_gradient_max_abs=float(ga.abs().max()),gradient_allclose=bool(torch.allclose(ga,gb,rtol=1e-5,atol=1e-8)))
            if values['output']!=0 or values['grid']!=0 or not values['gradient_allclose']:raise ValueError('Cache numeric bridge failed: '+str(values))
            differences.append(values)
        for repeat in range(6):
            order=['original','cached'] if repeat%2==0 else ['cached','original']
            for name in order:
                d=torch.zeros(4,128,device='cuda',requires_grad=True);torch.cuda.synchronize();start=time.perf_counter()
                a,_=condition_forward(model,lambda _a,_b:d,api,c,s) if name=='original' else prepared(d);torch.autograd.grad(a.square().mean(),d);torch.cuda.synchronize();timings[name].append(time.perf_counter()-start)
        report['cases'].append(dict(indices=[r['index'] for r in rows],prep_seconds=prep_seconds,exact_differences=differences,timings=timings));print(json.dumps(dict(images_checked=offset+4,total=64)),flush=True)
    if state_hash(model)!=before or any(p.grad is not None for p in model.parameters()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Frozen evidence changed')
    report.update(backbone_state_before=before,backbone_state_after=state_hash(model),finished=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,images=64,forward_exact=True,gradient_allclose=True,gradient_bitwise_exact_not_required=True)),flush=True)


if __name__=='__main__':main()
