"""Fixed six cross-source controls and exact current13 baseline bridge."""
import json
from pathlib import Path

from independent_coco import ROOT,digest


def main():
    import numpy as np
    import torch
    from PIL import Image
    from train_triplet_pilot import author_model
    from appearance_replication import initialize_replication_model
    from appearance_pairwise import appearance_forward
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    out=base/'ictone-coco-inputs-v1'
    out.mkdir(exist_ok=False)
    audit=ROOT/'outputs/adaptive-reference-20261113/ictone-cross-source-audit-v2/results.json'
    origin=json.loads(audit.read_text())
    assert origin['finished'] and origin['coco_exact_original_byte_matches']==[]
    assert digest(coco/'manifest.json')==origin['coco_manifest_sha256']
    jobs=json.loads((coco/'development/jobs.json').read_text())['jobs']
    fixed=[(785,785,'same_image'),(9483,19432,'global_contrast'),(872,1268,'global_desaturate'),
           (8021,8690,'region_color_luminance'),(285,1993,'region_tone_saturation'),(2149,2149,'same_image')]
    checkpoint=base/'multiref-control-v1/epoch-6.pth'
    assert digest(checkpoint)=='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    model,api,_=author_model()
    initialize_replication_model(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False)
    def read(path):
        with Image.open(path) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    def predict(content,reference):
        tensor=lambda im:(torch.from_numpy(np.array(im).copy()).permute(2,0,1)[None].float()/255).cuda()
        with torch.inference_mode():raw=appearance_forward(model,api,tensor(content),tensor(reference))[0]
        return Image.fromarray(np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8))
    bridge=base/'ictone-first-v1'
    before=predict(read(bridge/'content256.png'),read(bridge/'reference256.png'))
    with Image.open(base/'multiref-control-v1/epoch-6/22927.png') as im:
        assert np.array_equal(np.array(before),np.array(im)),'current13 baseline bridge mismatch'
    plan=dict(finished=False,scope='six consumed COCO development controls; no independent acceptance',
              origin_audit_sha256=digest(audit),checkpoint_sha256=digest(checkpoint),baseline_bridge_exact=True,
              source_sha256=digest(Path(__file__)),protocol_sha256=digest(ROOT/'docs/research/2026-11-17-ictone-coco.md'),cases=[])
    for ordinal,(target,reference,profile) in enumerate(fixed):
        content=read(coco/f'{target}-input.png')
        if profile=='same_image':
            ref=content.copy()
            truth_source=coco/f'{target}-input.png'
        else:
            job=next(x for x in jobs if x['target']==target and x['reference']==reference and x['profile']==profile)
            ref=read(coco/job['reference_file'])
            truth_source=coco/job['truth_file']
        identifier=f'{ordinal}-{target}'
        folder=out/identifier
        folder.mkdir()
        content.save(folder/'content.png')
        ref.save(folder/'reference.png')
        predict(content,ref).save(folder/'current13.png')
        spec=dict(index=identifier,content=dict(path=str(folder/'content.png'),sha256=digest(folder/'content.png')),
                  reference=dict(path=str(folder/'reference.png'),sha256=digest(folder/'reference.png')))
        (folder/'input-spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        plan['cases'].append(dict(index=identifier,ordinal=ordinal,target=target,reference=reference,profile=profile,
                                  spec_path=str(folder/'input-spec.json'),spec_sha256=digest(folder/'input-spec.json'),
                                  baseline_path=str(folder/'current13.png'),baseline_sha256=digest(folder/'current13.png'),
                                  truth_source=str(truth_source),truth_sha256=digest(truth_source),
                                  human_mask_source=str(coco/f'{target}-human-mask.png')))
    plan['finished']=True
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,cases=len(plan['cases']),baseline_bridge_exact=True)))


if __name__=='__main__':main()
