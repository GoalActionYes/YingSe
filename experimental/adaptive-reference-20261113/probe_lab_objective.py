"""Training-only gradient direction and scale probe, no model update."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from reference_normalization import distribution
from prepare_reference_normalization import state_hash
from lab_objective import lab,color_objective


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';load=lambda p:json.loads(p.read_text())
    assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    cohort=load(data/'cohort.json');rows=cohort['groups']['train'][:128];assert len({r['index'] for r in rows})==128
    manifest=data/'srgb-v1/manifest.json';assert digest(manifest)=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    lookup={e['original_path']:e for e in load(manifest)['files']}
    checkpoint=base/'training-coverage-expanded-v2/epoch-4.pth';assert digest(checkpoint)=='cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1'
    model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
    out=base/'lab-objective-probe-v1';out.mkdir(exist_ok=False)
    report=dict(scope='First128 consumed training rows; image-gradient diagnostic, no optimizer or dev selection',cohort_sha256=digest(data/'cohort.json'),manifest_sha256=digest(manifest),checkpoint_sha256=digest(checkpoint),source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['lab_objective.py','probe_lab_objective.py','lab-objective-probe-protocol.md','test_lab_objective.py']},cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    # Independent device paths, including near-black inputs.
    x=torch.tensor([0.,.001,.03,.04,.041,.3,.8,1.],dtype=torch.float64).reshape(1,1,2,4).repeat(1,3,1,1)
    x[:,1]=x[:,1].flip(-1);x[:,2]=x[:,2].flip(-2)
    values=[];gradients=[]
    for device in ['cpu','cuda']:
        t=x.to(device).requires_grad_();v=lab(t);g=torch.autograd.grad(v.square().mean(),t)[0];assert torch.isfinite(g).all();values.append(v.detach().cpu());gradients.append(g.detach().cpu())
    report['device_check']=dict(maximum_value_difference=float((values[0]-values[1]).abs().max()),maximum_gradient_difference=float((gradients[0]-gradients[1]).abs().max()))
    assert report['device_check']['maximum_value_difference']<1e-8 and report['device_check']['maximum_gradient_difference']<1e-8
    def read(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:a=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return (torch.from_numpy(a).permute(2,0,1)[None].float()/255).cuda(),hashlib.sha256(a.tobytes()).hexdigest()
    def cosine(a,b):
        a,b=a.flatten().double(),b.flatten().double();den=a.norm()*b.norm();assert den>0
        return float((a@b/den).detach())
    for i,row in enumerate(rows):
        pairs=[read(row[k]) for k in ['content','reference','gt']];c,s,gt=[e[0] for e in pairs]
        with torch.no_grad():p=appearance_forward(model,api,c,s)[0].clamp(0,1)
        p=p.detach().requires_grad_();l1=F.l1_loss(p,gt);mse=F.mse_loss(p,gt);candidate=color_objective(p,gt)
        g1=torch.autograd.grad(l1,p)[0];g0=torch.autograd.grad(mse,p)[0];g=torch.autograd.grad(candidate,p)[0]
        assert torch.isfinite(g).all();active=(g0.abs()>1e-12)&(g.abs()>1e-12)
        report['cases'].append(dict(index=row['index'],pixel_sha256={k:e[1] for k,e in zip(['content','reference','gt'],pairs)},
            rgb_mse=float(mse.detach()),color_loss=float(candidate.detach()),mse_lab_cosine=cosine(g0,g),l1_lab_cosine=cosine(g1,g),
            opposing_active_fraction=float(((g0*g<0)&active).sum()/active.sum()),rgb_gradient_mean_square=float(g0.double().square().mean()),color_gradient_mean_square=float(g.double().square().mean())))
        if (i+1)%32==0:print('checked',i+1,flush=True)
    rgb_rms=float(np.sqrt(np.mean([r['rgb_gradient_mean_square'] for r in report['cases']])));color_rms=float(np.sqrt(np.mean([r['color_gradient_mean_square'] for r in report['cases']])))
    report['summary']=dict(mean_mse_lab_cosine=float(np.mean([r['mse_lab_cosine'] for r in report['cases']])),mean_l1_lab_cosine=float(np.mean([r['l1_lab_cosine'] for r in report['cases']])),mean_opposing_active_fraction=float(np.mean([r['opposing_active_fraction'] for r in report['cases']])),rgb_gradient_rms=rgb_rms,color_gradient_rms=color_rms,frozen_color_scale=rgb_rms/color_rms)
    assert state_hash(model)==before
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(finished=True,model_state_unchanged=True,training=False,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-lab-objective-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(device=report['device_check'],summary=report['summary'])))


if __name__=='__main__':main()
