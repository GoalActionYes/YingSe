"""Frozen local pretraining descriptors, RGB-only, no optimizer or quality selection."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from dense_anchor_model import DenseAnchorModel


def state_hash(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';registry_path=b/'regional-correction-inputs-v1/results.json';r=json.loads(registry_path.read_text());assert r['finished'] and len(r['inputs'])==1365
    out=b/'semantic-token-inputs-v1';out.mkdir(exist_ok=False);(out/'tokens').mkdir()
    model=DenseAnchorModel();before=state_hash(model.model)
    report=dict(scope='Frozen pretrained patch cache, consumed RGB inputs only, no GT or training',registry_sha256=digest(registry_path),provenance=model.provenance,tf32=False,model_state_before=before,
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['prepare_semantic_tokens.py','semantic-token-preparation-protocol.md','dense_anchor_model.py']},inputs=[],training_started=False,finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    repeats=[]
    for item in r['inputs']:
        p=ROOT/item['path'];assert digest(p)==item['parent_sha256']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['pixel_sha256']
        features,colors,grid,meta=model.extract(rgb);h,w=grid;assert grid==(18,18)
        f=F.adaptive_avg_pool2d(features.T.reshape(1,384,h,w),(8,8)).flatten(2).transpose(1,2)[0]
        norms=f.norm(dim=1,keepdim=True);assert torch.isfinite(f).all() and (norms>1e-12).all()
        tokens=F.normalize(f,dim=1);local=F.adaptive_avg_pool2d(colors.T.reshape(1,3,h,w),(8,8)).flatten(2).transpose(1,2)[0]
        a,c=tokens.cpu().numpy(),local.cpu().numpy();assert a.shape==(64,384) and c.shape==(64,3) and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
        assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
        path=out/'tokens'/f"{item['index']:04d}.npz";np.savez_compressed(path,features=a,colors=c)
        report['inputs'].append(dict(index=item['index'],pixel_sha256=item['pixel_sha256'],path=str(path.relative_to(out)),sha256=digest(path),minimum_pre_normalization_norm=float(norms.min()),working_dimensions=meta['working_dimensions'],seconds=meta['seconds']))
        if item['index'] in (0,1364):
            f2,c2,g2,_=model.extract(rgb);assert torch.equal(features,f2) and torch.equal(colors,c2) and grid==g2;repeats.append(item['index'])
        if len(report['inputs'])%100==0:print(json.dumps(dict(prepared=len(report['inputs']),total=1365)),flush=True)
    assert state_hash(model.model)==before;report['model_state_after']=before;report['exact_endpoint_repeats']=repeats
    assert digest(registry_path)==report['registry_sha256'] and all(digest(Path(__file__).with_name(n))==v for n,v in report['source_sha256'].items())
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(finished=True,inputs=len(report['inputs']),exact_endpoint_repeats=repeats)),flush=True)


if __name__=='__main__':main()
