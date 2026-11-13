"""Synthetic function/gradient checks; never reads development images or GT."""
import argparse
import json
from pathlib import Path
import torch
from independent_coco import ROOT,digest
from tone_relations import source_tone_weights,tone_descriptor,tone_relation_losses


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='numeric-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/tone-relations'/args.run;out.mkdir(parents=True,exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['verify_tone_relations.py','tone_relations.py','tone-relations-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Synthetic descriptor/gradient and CPU/GPU numeric checks; no model quality, natural inputs or App acceptance',source_sha256={p.name:digest(p) for p in frozen.iterdir()},finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');torch.manual_seed(20261119);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.rand(2,3,4,4,dtype=torch.float64).repeat_interleave(2,0);truth=torch.rand_like(source);prediction=(torch.rand_like(source)*.6+.2).requires_grad_()
    function=lambda p:sum(w*v for w,v in zip([.25,.5],tone_relation_losses(p,truth,source)))
    if not torch.autograd.gradcheck(function,(prediction,),eps=1e-6,atol=1e-4,rtol=1e-3,raise_exception=True):raise ValueError('Finite difference failed')
    report['double_precision_gradcheck']=True
    if not torch.cuda.is_available():raise RuntimeError('GPU comparison is required for this developer check')
    report['cases']=[]
    for name in ['random','gray','black_white','low_chroma']:
        content=torch.rand(2,3,16,16).repeat_interleave(2,0)
        if name=='gray':content.fill_(.5)
        elif name=='black_white':content[:,:,:8]=0;content[:,:,8:]=1
        elif name=='low_chroma':content=content*.02+.49
        gt=(content*.75+.1).clone();gt[1]=torch.clamp(gt[1]+.15,0,1);gt[3,0]=torch.clamp(gt[3,0]-.1,0,1)
        results=[]
        for device in ['cpu','cuda']:
            c,g=content.to(device),gt.to(device);p=content.clone().to(device).requires_grad_();losses=tone_relation_losses(p,g,c)
            (.25*losses[0]+.5*losses[1]).backward()
            descriptor=tone_descriptor(p,source_tone_weights(c));gradient=p.grad
            if not torch.isfinite(descriptor).all() or not torch.isfinite(gradient).all():raise ValueError('Nonfinite synthetic result')
            zero=tone_relation_losses(g,g,c)
            if any(float(v)!=0 for v in zero):raise ValueError('Exact prediction not zero')
            results.append((descriptor.detach().cpu(),gradient.detach().cpu(),torch.tensor([float(v) for v in losses])))
        differences=[float((results[0][i]-results[1][i]).abs().max()) for i in range(3)]
        if max(differences)>1e-5:raise ValueError('CPU GPU divergence: '+str((name,differences)))
        report['cases'].append(dict(name=name,maximum_descriptor_difference=differences[0],maximum_gradient_difference=differences[1],maximum_loss_difference=differences[2],exact_prediction_zero=True,finite=True))
    report.update(finished=True,passed=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
