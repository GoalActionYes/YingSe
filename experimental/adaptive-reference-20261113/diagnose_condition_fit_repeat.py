"""Train-only multistart condition-label diagnosis; no product inference."""
import json,time,sys,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import condition_forward
from prepared_condition import PreparedCondition
from check_petal_condition import state_hash


def main():
    torch.set_num_threads(4);torch.manual_seed(20261126);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');features=load(base/'petal-condition-features-v1/results.json');bridge_path=base/'prepared-condition-bridge-v2/results.json';bridge=load(bridge_path)
    if not bridge['finished'] or digest(data/'cohort.json')!=features['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=features['prepared_manifest_sha256']:raise ValueError('Data/bridge changed')
    if any(v['output']!=0 or v['grid']!=0 or not v['gradient_allclose'] for c in bridge['cases'] for v in c['exact_differences']):raise ValueError('Cache bridge failed')
    for n,sha in bridge['source_sha256'].items():
        if digest(Path(__file__).with_name(n))!=sha:raise ValueError('Cache bridge source changed')
    rows=cohort['groups']['train'];groups={}
    for row in rows:groups.setdefault(row['group_id'],[]).append(row)
    if len(rows)!=1024 or len(groups)!=256 or any(sorted(r['style_index'] for r in g)!=list(range(4)) for g in groups.values()):raise ValueError('Wrong training split')
    rows=rows[:4]
    lookup={v['original_path']:v for v in prep['files']}
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    arrays={}
    for row in rows:
        for role in ['content','reference','gt']:
            entry=lookup[row[role]];p=data/entry['prepared_path']
            if digest(p)!=entry['sha256']:raise ValueError('Input changed')
            with Image.open(p) as im:arrays[row[role]]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    c,s,gt=[torch.from_numpy(np.stack([arrays[r[k]] for r in rows])).permute(0,3,1,2).float().cuda()/255 for k in ['content','reference','gt']];cached=PreparedCondition(model,api,c,s)
    gradients=[];outputs=[]
    for repeat in range(16):
        delta=torch.zeros(4,128,device='cuda',requires_grad=True);pred=cached(delta)[0].clamp(0,1);loss=F.l1_loss(pred,gt)+.1*(F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],gt[:,:,1:]-gt[:,:,:-1])+F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],gt[:,:,:,1:]-gt[:,:,:,:-1]));loss.backward();gradients.append(delta.grad.detach().cpu().numpy().copy());outputs.append(pred.detach().cpu().numpy().copy())
    delta=torch.zeros(4,128,device='cuda',requires_grad=True);pred=cached(delta)[0];loss=F.l1_loss(pred,gt);error=None
    torch.use_deterministic_algorithms(True)
    try:loss.backward()
    except RuntimeError as exc:error=str(exc)
    finally:torch.use_deterministic_algorithms(False)
    before_after=state_hash(model);result=dict(finished=True,source_sha256=digest(Path(__file__)),checkpoint_sha256=digest(cp),first_train_indices=[r['index'] for r in rows],cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),repeats=16,forward_max_abs=max(float(np.abs(o-outputs[0]).max()) for o in outputs),gradient_max_abs=max(float(np.abs(g-gradients[0]).max()) for g in gradients),gradient_exact_repeats=sum(np.array_equal(g,gradients[0]) for g in gradients),deterministic_backward_error=error,backbone_unchanged=before==before_after,scope='Same fixed CUDA forward/backward repeated, no optimization or product change')
    p=base/'condition-label-stability-v1'/('repeat-diagnostic.json' if len(sys.argv)==1 else sys.argv[1])
    if p.exists():raise ValueError('Refusing overwrite')
    p.write_text(json.dumps(result,indent=2),encoding='utf-8');np.savez_compressed(p.with_suffix('.npz'),gradients=np.stack(gradients));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
