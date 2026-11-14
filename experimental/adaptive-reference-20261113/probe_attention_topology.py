"""Actual current13 attention topology on synthetic inputs, not quality evidence."""
import json
from pathlib import Path
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';checkpoint=base/'multiref-control-v1/epoch-6.pth';expected='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    if digest(checkpoint)!=expected:raise ValueError('Current13 changed')
    out=base/'attention-topology-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_attention_topology.py','attention-topology-protocol.md','train_triplet_pilot.py','independent_coco.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sources={p.name:digest(p) for p in frozen.iterdir()};model,api,author=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    y,x=torch.meshgrid(torch.linspace(0,1,256,device='cuda'),torch.linspace(0,1,256,device='cuda'),indexing='ij');content=torch.stack([x,y,(x+y)/2])[None];reference=torch.stack([y,x,1-(x+y)/2])[None]
    with torch.no_grad():
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](content));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](reference));context=model.hist_projection(torch.cat([hc,hs],1));query=model.img_projection(api['TF'].resize(content,(64,64),antialias=True))
    query=query.detach().requires_grad_();context=context.detach();output=model.attention(query,context);weights=torch.linspace(.5,1.5,128,device='cuda')[None,:,None,None];loss=(output*weights).sum();qg,wg,kg=torch.autograd.grad(loss,[query,model.attention.q.weight,model.attention.kv.weight]);other=model.attention(-query.detach(),context).detach()
    result=dict(scope='Synthetic-input topology only, no image quality or learned semantic claim',checkpoint_sha256=expected,source_sha256=sources,author_revision=author['revision'],author_model_source_sha256=digest(ROOT/'.tools/Hist2Style-source/demo/model.py'),query_shape=list(query.shape),context_shape=list(context.shape),output_shape=list(output.shape),spatial_output_maximum_difference=float((output-output[:,:,:1,:1]).abs().max()),negated_query_output_maximum_difference=float((output.detach()-other).abs().max()),query_gradient_maximum=float(qg.abs().max()),q_weight_gradient_maximum=float(wg.abs().max()),key_weight_gradient_maximum=float(kg[:128].abs().max()),value_weight_gradient_maximum=float(kg[128:].abs().max()),finished=True,app_promoted=False)
    if any(not torch.isfinite(t).all() for t in [output,other,qg,wg,kg]) or digest(checkpoint)!=expected or any(digest(Path(__file__).with_name(n))!=sha for n,sha in sources.items()):raise ValueError('Nonfinite probe or changed frozen evidence')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
