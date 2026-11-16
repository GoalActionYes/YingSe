"""Check all64 development zero-head outputs against frozen native RGB8 baseline."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from regional_correction import RegionalCorrection,global_control


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';cache=base/'regional-correction-inputs-v1'
    r=json.loads((cache/'results.json').read_text());assert r['finished']
    assert digest(cache/'descriptors.npz')==r['descriptors_sha256']
    inputs={x['index']:x for x in r['inputs']}
    with np.load(cache/'descriptors.npz') as a:features=a['regional']
    out=base/'regional-correction-bridge-v1';out.mkdir(exist_ok=False)
    engine=SpatialEngine('multiref-v1');torch.manual_seed(20261127);head=RegionalCorrection().to(engine.device)
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    rows=[]
    def image(item):
        p=ROOT/item['path'];assert digest(p)==item['parent_sha256']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['pixel_sha256'];return rgb
    with torch.inference_mode():
        for row in r['rows']:
            if row['split']!='validation':continue
            a,b=[inputs[row['roles'][role]['feature_index']] for role in ['content','reference']]
            rgb,ref=image(a),image(b);assert digest(cache/a['mask_path'])==a['mask_sha256']
            p=torch.from_numpy(np.load(cache/a['mask_path'])[None,None]).to(engine.device)
            x=engine.tensor(rgb);prepared=engine.prepare(rgb,ref)
            baseline=prepared.float_output(rgb);native=prepared.byte_output(rgb)
            raw=torch.from_numpy(baseline).permute(2,0,1)[None].to(engine.device)
            descriptors=[torch.from_numpy(features[item['index']][None]).to(engine.device) for item in [a,b]]
            for mode in ['regional','control']:
                d=descriptors if mode=='regional' else [global_control(v) for v in descriptors]
                actual=head(x,raw,p,*d)
                assert torch.equal(actual,raw)
                q=np.rint(actual[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                assert np.array_equal(q,native)
                assert torch.equal(head(x,raw,p,*d,enabled=False),raw)
            rows.append(dict(index=row['index'],pixel_sha256=hashlib.sha256(native.tobytes()).hexdigest()))
    assert len(rows)==64 and all(torch.equal(v,initial[k]) for k,v in head.state_dict().items())
    report=dict(scope='Zero-initialized regional correction bridge, not trained quality',input_report_sha256=digest(cache/'results.json'),model_sha256=engine.checkpoint_sha,
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['regional_correction.py','check_regional_correction.py','regional-correction-training-protocol.md']},
        rows=rows,regional_and_control_exact_float=True,native_rgb8_exact=True,disabled_exact=True,head_state_unchanged=True,training_started=False,finished=True)
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
