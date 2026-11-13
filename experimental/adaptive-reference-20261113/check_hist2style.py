"""Author spatial bilateral-grid inference on consumed development only."""
import ast
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor,to_pil_image
from independent_coco import ROOT,digest


def load(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def small(rgb):
    im=Image.fromarray(rgb);im.thumbnail((640,640),Image.Resampling.BILINEAR);return np.array(im)


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'.tools/Hist2Style-source'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision']!='a3ef9ba0455d957b54c459c88cac36cf5740c164':raise ValueError('Unexpected revision')
    for entry in manifest['files']:
        if digest(source/entry['path'])!=entry['sha256']:raise ValueError('Asset changed')
    code=ast.parse((source/'demo/model.py').read_text(encoding='utf-8'))
    omitted=[];body=[]
    for node in code.body:
        if isinstance(node,ast.Import) and any(x.name=='lightning' for x in node.names):omitted.append('lightning import')
        elif isinstance(node,ast.ClassDef) and node.name=='Model':omitted.append('Lightning Model wrapper')
        else:body.append(node)
    if omitted!=['lightning import','Lightning Model wrapper']:raise ValueError('Unexpected AST adapter')
    code.body=body;api={'__name__':'hist2style_author_inference'}
    exec(compile(code,str(source/'demo/model.py'),'exec'),api)
    model=api['Grid']()
    payload=torch.load(source/'demo/model-epoch=1500-val_loss=2.79.ckpt',map_location='cpu',weights_only=True)
    original=payload['state_dict']
    if not all(k.startswith('model.') for k in original):raise ValueError('Unexpected weight namespace')
    state={k.removeprefix('model.'):v for k,v in original.items()}
    expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('Missing/extra inference state')
    for k,v in state.items():
        if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not bool(torch.isfinite(v).all()):raise ValueError('Invalid state: '+k)
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False).to('cuda')
    data=ROOT/'outputs/adaptive-reference-20261113/pst50';baseline=data/'development-canonical-final-v1'
    out=ROOT/'outputs/adaptive-reference-20261113/hist2style/development-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['check_hist2style.py','prepare_hist2style.py','hist2style-protocol.md']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    rows=[]
    report=dict(scope='Consumed development 1-15; no unseen acceptance or product integration',author=manifest,
        omitted_ast_nodes=omitted,strict_state_keys=len(state),source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=rows)
    def tensor(rgb):return pil_to_tensor(Image.fromarray(rgb))[None].to('cuda').float()/255
    @torch.inference_mode()
    def predict(target,reference):
        start=time.perf_counter();c,s=tensor(target),tensor(reference);h,w=c.shape[2:]
        if h*w>2000000:
            scale=math.sqrt(2000000/(h*w));c=F.interpolate(c,size=(round(h*scale),round(w*scale)),mode='bilinear',align_corners=False,antialias=True)
        hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s))
        torch.cuda.synchronize();prepared=time.perf_counter()
        output,grid=model(c,hc,hs);torch.cuda.synchronize();inferred=time.perf_counter()
        if not bool(torch.isfinite(output).all()):raise ValueError('Nonfinite output')
        rgb=np.array(to_pil_image(output[0].clamp(0,1).cpu()))
        return rgb,dict(preprocess_seconds=prepared-start,inference_seconds=inferred-prepared,output_shape=list(rgb.shape),grid_shape=list(grid.shape))
    for i in range(1,16):
        inputs=data if i<=5 else data/'development-dense-round2-v1/inputs'
        a,b=inputs/f'content_709/in{i}.png',inputs/f'paired_style/tar{i}.png'
        rgb,timing=predict(load(a),load(b));path=out/f'{i}-output.png';Image.fromarray(rgb).save(path)
        row=dict(index=i,timing=timing,sha256=digest(path),source_sha256=dict(target=digest(a),reference=digest(b)));rows.append(row)
        print(json.dumps(row),flush=True)
    same=load(data/'content_709/in1.png');identity,it=predict(same,same);Image.fromarray(identity).save(out/'identity.png')
    report['identity_timing']=it
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        i=row['index'];inputs=data if i<=5 else data/'development-dense-round2-v1/inputs'
        if digest(out/f'{i}-output.png')!=row['sha256']:raise ValueError('Prediction changed')
        gt=small(load(inputs/f'paired_gt/gt{i}.png'));prediction=small(load(out/f'{i}-output.png'))
        images={k:load(baseline/f'{i}-{k}.png') for k in ['input','current','final']};images['hist2style']=prediction
        row['scores']={k:float(np.abs(v.astype(np.float32)-gt).mean()) for k,v in images.items()}
        panel=Image.new('RGB',(1080,520),'#222222');draw=ImageDraw.Draw(panel)
        entries=[('Input',images['input']),('Reference',small(load(inputs/f'paired_style/tar{i}.png'))),('Expert',gt),('Current',images['current']),('Canon final',images['final']),('Hist2Style',prediction)]
        for n,(label,rgb) in enumerate(entries):
            im=Image.fromarray(rgb);im.thumbnail((356,230));x,y=n%3*360,n//3*260
            panel.paste(im,(x+(360-im.width)//2,y+25));draw.text((x+8,y+5),label,fill='white')
        panel.save(out/f'{i}-comparison.jpg',quality=94)
    before=np.array([r['scores']['current'] for r in rows]);summary={}
    for name in ['input','current','final','hist2style']:
        a=np.array([r['scores'][name] for r in rows]);summary[name]=dict(mean=float(a.mean()),improved=int((a<before-.1).sum()),regressed=int((a>before+.1).sum()),severe=int(((a>before+5)&(a>before*1.25)).sum()))
    report.update(summary=summary,predictions_sha256=digest(out/'predictions.json'),all_predictions_before_expert_read=True,
        raw_same_image_mae=float(np.abs(small(identity).astype(np.float32)-small(same)).mean()),product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=summary,raw_same_image_mae=report['raw_same_image_mae'])),flush=True)


if __name__=='__main__':main()
