"""Isolated class-conditioned relative-color residual, no manual masks."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
from independent_coco import ROOT,digest


def load(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def small(rgb):
    im=Image.fromarray(rgb);im.thumbnail((640,640),Image.Resampling.BILINEAR);return np.array(im)


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assets=ROOT/'.tools/scene-segformer';manifest=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        if digest(assets/entry['path'])!=entry['sha256']:raise ValueError('Asset mismatch')
    processor=SegformerImageProcessor.from_pretrained(assets,local_files_only=True)
    model,info=SegformerForSemanticSegmentation.from_pretrained(assets,local_files_only=True,use_safetensors=True,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Incomplete model restoration')
    model.eval().requires_grad_(False).to('cuda')
    out=ROOT/'outputs/adaptive-reference-20261113/scene-regions/development-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_scene_regions.py','prepare_scene_segmentation.py','scene-region-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    @torch.inference_mode()
    def segment(rgb):
        start=time.perf_counter();inputs=processor(images=rgb,return_tensors='pt').to('cuda');logits=model(**inputs).logits
        probability=F.interpolate(logits,size=rgb.shape[:2],mode='bilinear',align_corners=False).softmax(1)[0]
        confidence,label=probability.max(0);torch.cuda.synchronize()
        return probability,label.cpu().numpy(),confidence.cpu().numpy(),time.perf_counter()-start
    data=ROOT/'outputs/adaptive-reference-20261113/pst50';prior=data/'development-canonical-final-v1';rows=[]
    report=dict(scope='Consumed development only, not semantic/video/product acceptance',model=manifest,loading_info=info,cases=rows,source_sha256={p.name:digest(p) for p in frozen.iterdir()})
    for i in range(1,16):
        source=data if i<=5 else data/'development-dense-round2-v1/inputs';a,b=source/f'content_709/in{i}.png',source/f'paired_style/tar{i}.png'
        target,reference=small(load(a)),small(load(b));base=load(prior/f'{i}-current.png')
        probability,label,confidence,t1=segment(target);_,rlabel,rconf,t2=segment(reference)
        np.savez_compressed(out/f'{i}-segmentation.npz',target_label=label,target_confidence=confidence,reference_label=rlabel,reference_confidence=rconf)
        supported=[];deltas=[]
        for category in sorted(set(np.unique(label))&set(np.unique(rlabel))):
            mask=(label==category)&(confidence>=.7);rmask=(rlabel==category)&(rconf>=.7)
            if mask.mean()<.015 or rmask.mean()<.015:continue
            delta=np.median(reference[rmask].astype(float)/255,0)-np.median(base[mask].astype(float)/255,0)
            supported.append(dict(id=int(category),name=model.config.id2label[int(category)],target_support=float(mask.mean()),reference_support=float(rmask.mean()),raw_delta=delta.tolist()));deltas.append(delta)
        correction=torch.zeros((3,*base.shape[:2]),device='cuda')
        if len(supported)>=2:
            centered=np.array(deltas)-np.median(deltas,axis=0)
            for entry,delta in zip(supported,centered):
                delta=np.clip(delta,-.12,.12);entry['applied_relative_delta']=delta.tolist()
                correction+=probability[entry['id']][None]*torch.tensor(delta,device='cuda',dtype=torch.float32)[:,None,None]
        change=correction.permute(1,2,0).cpu().numpy()
        result=np.rint(np.clip(base.astype(float)/255+.5*change,0,1)*255).astype(np.uint8)
        path=out/f'{i}-output.png';Image.fromarray(result).save(path)
        rows.append(dict(index=i,sha256=digest(path),source_sha256=dict(target=digest(a),reference=digest(b)),supported=supported,applied=len(supported)>=2,segmentation_seconds=[t1,t2]))
        print(json.dumps(dict(index=i,classes=[x['name'] for x in supported])),flush=True)
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        i=row['index'];source=data if i<=5 else data/'development-dense-round2-v1/inputs'
        if digest(out/f'{i}-output.png')!=row['sha256']:raise ValueError('Prediction changed')
        truth=small(load(source/f'paired_gt/gt{i}.png'));current=load(prior/f'{i}-current.png');candidate=load(out/f'{i}-output.png')
        row['scores']={k:float(np.abs(v.astype(np.float32)-truth).mean()) for k,v in [('current',current),('regional',candidate)]}
        panel=Image.new('RGB',(1080,520),'#222222');draw=ImageDraw.Draw(panel)
        for n,(label,rgb) in enumerate([('Input',small(load(source/f'content_709/in{i}.png'))),('Reference',small(load(source/f'paired_style/tar{i}.png'))),('Expert',truth),('Current',current),('Semantic relative color',candidate)]):
            im=Image.fromarray(rgb);im.thumbnail((356,230));x,y=n%3*360,n//3*260;panel.paste(im,(x,y+25));draw.text((x+8,y+5),label,fill='white')
        panel.save(out/f'{i}-comparison.jpg',quality=94)
    baseline=np.array([r['scores']['current'] for r in rows]);actual=np.array([r['scores']['regional'] for r in rows])
    report.update(summary=dict(count=15,applied=sum(r['applied'] for r in rows),current_mean=float(baseline.mean()),regional_mean=float(actual.mean()),improved=int((actual<baseline-.1).sum()),regressed=int((actual>baseline+.1).sum()),severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum())),predictions_sha256=digest(out/'predictions.json'),all_predictions_before_expert_read=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
