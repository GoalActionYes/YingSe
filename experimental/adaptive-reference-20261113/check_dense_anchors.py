"""First-five real development pairs; all predictions precede expert-target reads."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from canonical_model import CanonicalModel,ROOT
from independent_coco import digest
from dense_anchor_model import DenseAnchorModel
from dense_anchors import match_anchors,color_from_anchors,choose_blend


def load(path,size=None):
    with Image.open(path) as image:
        image=image.convert('RGB')
        if size: image=image.resize(size,Image.Resampling.BILINEAR)
        else: image.thumbnail((640,640),Image.Resampling.BILINEAR)
        return np.array(image)


def tensor(rgb,device): return torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device,dtype=torch.float32)/255


def rgb8(t): return t[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='development-dense-anchors-native-v1'); args=parser.parse_args()
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'; out=data/args.out; out.mkdir(exist_ok=False)
    names=('dense_anchors.py','dense_anchor_model.py','check_dense_anchors.py','dense-anchor-protocol.md','prepare_dense_anchor.py','canonical_model.py')
    frozen=out/'frozen-source'; frozen.mkdir()
    hashes={}
    for name in names:
        path=Path(__file__).with_name(name); (frozen/name).write_bytes(path.read_bytes()); hashes[name]=digest(path)
    report=dict(scope='Already-used PST50 development1-5 and raw same-image; not independent acceptance',source_sha256=hashes,cases=[])
    result=out/'results.json'
    def save(): result.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save(); torch.set_num_threads(4)
    start=time.perf_counter(); model=DenseAnchorModel(); canonical=CanonicalModel()
    report.update(load_seconds=time.perf_counter()-start,descriptor=model.provenance,canonical=canonical.provenance)
    save()
    for index in range(1,7):
        same=index==6; n=1 if same else index
        target_path=data/f'content_709/in{n}.png'; reference_path=target_path if same else data/f'paired_style/tar{n}.png'
        target,reference=load(target_path),load(reference_path)
        size=(target.shape[1],target.shape[0])
        base=target if same else load(data/f'development-salut/{n}-Current.png',size)
        # Model grading must be inferred from the original decoded images.
        # The separate resolution diagnostic preserves and explains the old trial.
        with Image.open(target_path) as im: original_target=np.array(im.convert('RGB'))
        with Image.open(reference_path) as im: original_reference=np.array(im.convert('RGB'))
        full_canon,canon_time=canonical.predict(original_target,original_reference)
        canon=np.array(Image.fromarray(full_canon).resize(size,Image.Resampling.BILINEAR))
        features,colors,grid,target_time=model.extract(target)
        ref_features,ref_colors,_,reference_time=model.extract(reference)
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.inference_mode():
            anchors,confidence=match_anchors(features,ref_features,ref_colors)
            anchors=anchors.T.reshape(1,3,*grid); confidence=confidence.reshape(1,1,*grid)
            global_result,global_diag=color_from_anchors(tensor(base,model.device),anchors,confidence)
            spatial_result,spatial_diag=color_from_anchors(tensor(base,model.device),anchors,confidence,spatial=True)
            blend,blend_diag=choose_blend(tensor(base,model.device),tensor(canon,model.device),anchors,confidence)
        torch.cuda.synchronize(); render_seconds=time.perf_counter()-started
        outputs=dict(current=base,canonical=canon,anchor_global=rgb8(global_result),anchor_spatial=rgb8(spatial_result),anchor_blend=rgb8(blend))
        for name,image in outputs.items(): Image.fromarray(image).save(out/f'{index}-{name}.png')
        # No desired image/score has been consulted by features, fit, or blend selection.
        truth=target if same else load(data/f'paired_gt/gt{n}.png',size)
        rowscores={name:float(np.abs(image.astype(np.float32)-truth.astype(np.float32)).mean()) for name,image in dict(input=target,**outputs).items()}
        row=dict(index=index,same_image=same,source_sha256=dict(target=digest(target_path),reference=digest(reference_path)),
            shape=list(target.shape),scores=rowscores,global_diagnostics=global_diag,spatial_diagnostics=spatial_diag,
            blend=blend_diag,feature_time=dict(target=target_time,reference=reference_time),canonical_time=canon_time,
            fit_and_application_seconds=render_seconds,
            timing_scope='Includes tensor upload and small output setup; source/descriptor processing, base mapping, I/O separate')
        report['cases'].append(row); save(); print(json.dumps(row),flush=True)
        conf=F.interpolate(confidence,size=target.shape[:2],mode='bilinear',align_corners=False).repeat(1,3,1,1)
        conf_rgb=rgb8(conf)
        panel=Image.new('RGB',(1080,780),'#222222'); draw=ImageDraw.Draw(panel)
        images=[('Input',target),('Reference',reference),('Expert target',truth),('Current',base),('Canon',canon),
            ('Anchor chosen blend',outputs['anchor_blend']),('Anchor global',outputs['anchor_global']),
            ('Anchor spatial',outputs['anchor_spatial']),('Confidence',conf_rgb)]
        for k,(label,image) in enumerate(images):
            im=Image.fromarray(image); im.thumbnail((356,232)); x,y=k%3*360,k//3*260
            panel.paste(im,(x+(360-im.width)//2,y+24+(232-im.height)//2)); draw.text((x+8,y+6),label,fill='white')
        panel.save(out/f'{index}-comparison.jpg',quality=94)


if __name__=='__main__': main()
