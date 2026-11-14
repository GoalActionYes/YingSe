"""Native original-encoder bridge on synthetic inputs only."""
import json
from pathlib import Path
import torch
from independent_coco import ROOT,digest
from canonical_model import CanonicalModel,CHECKPOINT_SHA
from frozen_grade_encoder import load_frozen_grade_encoder


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'frozen-grade-bridge-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_frozen_grade_bridge.py','frozen_grade_encoder.py','test_frozen_grade_encoder.py','frozen-grade-diagnostic-protocol.md','canonical_model.py','training_checks.py','independent_coco.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    original=CanonicalModel(device='cuda');wrapped=load_frozen_grade_encoder('cuda');a=original.model.Embedding_Net.state_dict();b=wrapped.encoder.state_dict()
    if set(a)!=set(b) or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Encoder state differs')
    before={k:v.clone() for k,v in b.items()};wrapped.train(True);cases=[]
    with torch.no_grad():
        for batch,h,w in [(1,256,256),(2,173,281),(1,720,1280)]:
            y,x=torch.meshgrid(torch.linspace(0,1,h,device='cuda'),torch.linspace(0,1,w,device='cuda'),indexing='ij');rgb=torch.stack([x,y,1-(x+y)/2])[None].repeat(batch,1,1,1);expected=original.model.Embedding_Net(rgb);actual=wrapped(rgb);difference=float((expected-actual).abs().max())
            if difference!=0:raise ValueError('Native original grade vectors differ')
            cases.append(dict(shape=list(rgb.shape),vector_shape=list(actual.shape),maximum_float_difference=difference,minimum_norm=float(actual.norm(dim=1).min()),maximum_norm=float(actual.norm(dim=1).max())))
    if wrapped.training or any(p.requires_grad for p in wrapped.parameters()) or any(not torch.equal(v,wrapped.encoder.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen encoder state changed')
    sources={p.name:digest(p) for p in frozen.iterdir()}
    if digest(ROOT/'.tools/CanonCGT/pretrained/E2E.pth')!=CHECKPOINT_SHA or any(digest(Path(__file__).with_name(k))!=v for k,v in sources.items()):raise ValueError('Bridge sources/assets changed')
    report=dict(scope='Synthetic native original grade-encoder parity only; no quality acceptance',source_sha256=sources,provenance=wrapped.provenance,state_tensors_identical=len(a),cases=cases,frozen_buffers_unchanged=True,training_mode_remains_false=True,finished=True,passed=True,app_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','provenance']}),flush=True)


if __name__=='__main__':main()
