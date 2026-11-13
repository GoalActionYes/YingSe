"""Isolate repeated CUDA control updates and 3D grid-sample backward noise."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import enable_pairwise_context
from train_consistent_reference import forward
from paired_training import grouped_batches

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';control_dir=base/'consistent-reference-control-v1';out=base/'appearance-cuda-roundoff-v1';out.mkdir(exist_ok=False)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));control=load(control_dir/'results.json');cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={r['original_path']:r for r in prep['files']}
    report=dict(scope='Training-numerics diagnosis only; no candidate/dev selection',torch_version=torch.__version__,source_sha256=digest(Path(__file__)),control_results_sha256=digest(control_dir/'results.json'),finished=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    batch=[cohort['groups']['train'][i] for i in grouped_batches(cohort['groups']['train'],np.random.default_rng(20261122))[0]]
    def tensor(relative):
        item=lookup[relative];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('Changed training input')
        with Image.open(p) as im:x=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return (torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda()
    c,s,gt=[torch.cat([tensor(r[role]) for r in batch]) for role in ['content','reference','gt']]
    model,api,_=author_model();enable_pairwise_context(model);initial=torch.load(control_dir/'epoch-0.pth',map_location='cpu',weights_only=True);states=[];runs=[]
    for repeat in range(3):
        model.load_state_dict(initial,strict=True);model.train();torch.manual_seed(20261122);torch.cuda.manual_seed_all(20261122)
        extra=list(model.pairwise_adapter.parameters());ids={id(p) for p in extra};optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=1e-5),dict(params=extra,lr=1e-4)],lr=1e-5,weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True);prediction=forward(model,api,c,s)[0].clamp(0,1);loss=F.l1_loss(prediction,gt)+.1*(F.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(prediction[:,:,:,1:]-prediction[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]));loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});runs.append(dict(loss=float(loss),gradient_norm=float(norm)))
    report['same_control_repeated_steps']=runs;report['same_control_parameter_differences']=[max(float((a[k]-b[k]).abs().max()) for k in a) for a,b in [(states[0],states[1]),(states[0],states[2])]]
    torch.manual_seed(20261123);source=torch.randn(1,13,8,16,16,device='cuda');coords=torch.rand(1,1,256,256,3,device='cuda')*2-1;weight=torch.randn(1,13,1,256,256,device='cuda');gradients=[];outputs=[]
    for repeat in range(3):
        x=source.detach().clone().requires_grad_();y=F.grid_sample(x,coords,padding_mode='border',align_corners=False);(y*weight).sum().backward();outputs.append(y.detach());gradients.append(x.grad.detach())
    report['isolated_grid_sample']=dict(forward_max_difference=max(float((outputs[0]-x).abs().max()) for x in outputs[1:]),backward_max_difference=max(float((gradients[0]-x).abs().max()) for x in gradients[1:]))
    torch.use_deterministic_algorithms(True)
    try:
        x=source.detach().clone().requires_grad_();F.grid_sample(x,coords,padding_mode='border',align_corners=False).sum().backward()
    except RuntimeError as exc:report['strict_determinism_error']=str(exc)
    else:report['strict_determinism_error']=None
    finally:torch.use_deterministic_algorithms(False)
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
