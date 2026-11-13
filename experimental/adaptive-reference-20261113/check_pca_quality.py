"""Freeze 15 consumed-development predictions, then score against expert targets."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from pca_kd import PCAKD


def load(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))


def small(rgb):
    image=Image.fromarray(rgb);image.thumbnail((640,640),Image.Resampling.BILINEAR)
    return np.array(image)


def tensor(rgb,device,long_edge=640):
    h,w=rgb.shape[:2];scale=min(1,long_edge/max(h,w));size=(int(h*scale)//8*8,int(w*scale)//8*8)
    x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native',action='store_true');args=parser.parse_args()
    long_edge=100000 if args.native else 640
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    previous=data/'development-canonical-final-v1'
    out=ROOT/'outputs/adaptive-reference-20261113/pca-kd'/('development-native-v1' if args.native else 'development-v1');out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['pca_kd.py','check_pca_quality.py','postprocess_pca.py','pca-kd-protocol.md']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    model=PCAKD(device='cuda');rows=[]
    report=dict(scope='Consumed development 1-15 only; not independent acceptance or video validation',
        preprocessing='Native dimensions floored to multiples of eight' if args.native else 'Whole-image longest side 640, floored to multiples of eight',
        converted_manifest_sha256=digest(ROOT/'.tools/PCA-KD-converted/manifest.json'),
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=rows)
    for index in range(1,16):
        source=data if index<=5 else data/'development-dense-round2-v1/inputs'
        target,style=load(source/f'content_709/in{index}.png'),load(source/f'paired_style/tar{index}.png')
        start=time.perf_counter();c,s=tensor(target,'cuda',long_edge),tensor(style,'cuda',long_edge)
        torch.cuda.synchronize();prepared=time.perf_counter()
        output=model(c,s);torch.cuda.synchronize();inferred=time.perf_counter()
        rgb=output[0].permute(1,2,0).cpu().numpy()
        Image.fromarray((rgb*255).astype(np.uint8)).save(out/f'{index}-raw.png')
        np.savez_compressed(out/f'{index}-raw.npz',rgb=rgb,guide=target)
        rows.append(dict(index=index,preprocess_seconds=prepared-start,inference_seconds=inferred-prepared,
            source_sha256=dict(target=digest(source/f'content_709/in{index}.png'),reference=digest(source/f'paired_style/tar{index}.png'))))
        print(json.dumps(rows[-1]),flush=True)
    # Raw same-image behavior is measured, never concealed by an identity guard.
    same=load(data/'content_709/in1.png');c=tensor(same,'cuda',long_edge)
    raw=model(c,c)[0].permute(1,2,0).cpu().numpy()
    np.savez_compressed(out/'identity-raw.npz',rgb=raw,guide=same)
    Image.fromarray((raw*255).astype(np.uint8)).save(out/'identity-raw.png')
    result=subprocess.run([str(ROOT/'.tools/tf-convert-env/Scripts/python.exe'),str(Path(__file__).with_name('postprocess_pca.py')),str(out)],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    (out/'smoothing.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Author smoothing failed; see smoothing.log')
    report['prediction_hashes']={p.name:digest(p) for p in out.glob('*.png')}
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Only now read expert targets. No case-specific strength or branch selection.
    for row in rows:
        i=row['index'];source=data if i<=5 else data/'development-dense-round2-v1/inputs'
        truth=small(load(source/f'paired_gt/gt{i}.png'))
        images={k:load(previous/f'{i}-{k}.png') for k in ['input','current','final']}
        images.update({k:small(load(out/f'{i}-{k}.png')) for k in ['raw','guided']})
        row['scores']={k:float(np.abs(v.astype(np.float32)-truth).mean()) for k,v in images.items()}
        panel=Image.new('RGB',(1080,780),'#222222');draw=ImageDraw.Draw(panel)
        all_images=[('Input',images['input']),('Reference',small(load(source/f'paired_style/tar{i}.png'))),('Expert',truth),
            ('Current',images['current']),('Canon final',images['final']),('PCA raw',images['raw']),('PCA guided',images['guided'])]
        for n,(label,rgb) in enumerate(all_images):
            im=Image.fromarray(rgb);im.thumbnail((356,230));x,y=n%3*360,n//3*260
            panel.paste(im,(x+(360-im.width)//2,y+25));draw.text((x+8,y+5),label,fill='white')
        panel.save(out/f'{i}-comparison.jpg',quality=94)
    summary={}
    baseline=np.array([r['scores']['current'] for r in rows])
    for k in ['input','current','final','raw','guided']:
        actual=np.array([r['scores'][k] for r in rows]);summary[k]=dict(mean=float(actual.mean()),
            improved=int((actual<baseline-.1).sum()),regressed=int((actual>baseline+.1).sum()),
            severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum()))
    original=small(same).astype(np.float32)
    report.update(summary=summary,predictions_sha256=digest(out/'predictions.json'),
        raw_same_image_mae={k:float(np.abs(small(load(out/f'identity-{k}.png')).astype(np.float32)-original).mean()) for k in ['raw','guided']},
        all_predictions_before_expert_read=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,raw_same_image_mae=report['raw_same_image_mae'])),flush=True)


if __name__=='__main__':main()
