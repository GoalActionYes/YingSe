"""Fixed final-weight comparison on fifteen consumed development pairs only."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from canonical_model import ROOT,CanonicalModel
from canonical_final import CanonicalFinal
from independent_coco import digest
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze,algorithm_fingerprint
from reference_color_app.lut import apply_mapping


def load(path):
    with Image.open(path) as image: return np.array(image.convert('RGB'))


def small(rgb):
    image=Image.fromarray(rgb);image.thumbnail((640,640),Image.Resampling.BILINEAR);return np.array(image)


def main():
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    out=data/'development-canonical-final-v1';out.mkdir(exist_ok=False)
    names=('canonical_final.py','prepare_canonical_final.py','canonical-final-protocol.md','check_canonical_final.py','canonical_model.py')
    frozen=out/'frozen-source';frozen.mkdir(); hashes={}
    for name in names:
        path=Path(__file__).with_name(name);(frozen/name).write_bytes(path.read_bytes());hashes[name]=digest(path)
    torch.set_num_threads(4);model=CanonicalFinal();e2e=CanonicalModel();rows=[]
    report=dict(scope='PST50 1-15 consumed development, no heldout use or product integration',
        source_sha256=hashes,current_algorithm_fingerprint=algorithm_fingerprint(),final=model.provenance,e2e=e2e.provenance,cases=rows)
    for index in range(1,16):
        source=data if index<=5 else data/'development-dense-round2-v1/inputs'
        target_path=source/f'content_709/in{index}.png';reference_path=source/f'paired_style/tar{index}.png'
        row=dict(index=index,source_sha256=dict(target=digest(target_path),reference=digest(reference_path)),outputs={});rows.append(row)
        try:
            target,reference=load(target_path),load(reference_path)
            start=time.perf_counter(); current=small(apply_mapping(target,analyze(target_path,reference_path),1)); row['current_seconds']=time.perf_counter()-start
            before,t1=e2e.predict(target,reference); final,t2=model.predict(target,reference)
            output=dict(input=small(target),current=current,e2e=small(before),final=small(final))
            for name,rgb in output.items():
                path=out/f'{index}-{name}.png';Image.fromarray(rgb).save(path);row['outputs'][name]=dict(path=path.name,sha256=digest(path))
            row.update(timing=dict(e2e=t1,final=t2),shape=list(target.shape))
        except Exception as exc:row['failure']=dict(type=type(exc).__name__,message=str(exc))
        (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(index=index,failure=row.get('failure'),timing=row.get('timing'))),flush=True)
    # All predictions above are frozen before reading expert output images here.
    for row in rows:
        if 'failure' in row:continue
        index=row['index'];source=data if index<=5 else data/'development-dense-round2-v1/inputs'
        truth=small(load(source/f'paired_gt/gt{index}.png'))
        images={name:load(out/entry['path']) for name,entry in row['outputs'].items()}
        row['scores']={name:float(np.abs(rgb.astype(np.float32)-truth.astype(np.float32)).mean()) for name,rgb in images.items()}
        panel=Image.new('RGB',(1080,520),'#222222');draw=ImageDraw.Draw(panel)
        for k,(label,rgb) in enumerate([('Input',images['input']),('Reference',small(load(source/f'paired_style/tar{index}.png'))),('Expert target',truth),
            ('Current',images['current']),('Canon E2E',images['e2e']),('Canon final SSL',images['final'])]):
            im=Image.fromarray(rgb);im.thumbnail((356,232));x,y=k%3*360,k//3*260
            panel.paste(im,(x+(360-im.width)//2,y+24+(232-im.height)//2));draw.text((x+8,y+6),label,fill='white')
        panel.save(out/f'{index}-comparison.jpg',quality=94)
    valid=[x for x in rows if 'scores' in x];summary={}
    for name in ('input','current','e2e','final'):
        a=np.array([x['scores'][name] for x in valid]);b=np.array([x['scores']['current'] for x in valid])
        summary[name]=dict(count=len(valid),failed=len(rows)-len(valid),mean=float(a.mean()) if len(a) else None,
            median=float(np.median(a)) if len(a) else None,improved=int((a<b-.1).sum()),regressed=int((a>b+.1).sum()),
            severe=int(((a>b+5)&(a>b*1.25)).sum()))
    source=load(data/'content_709/in1.png');raw,raw_time=model.predict(source,source)
    report.update(summary=summary,raw_same_image=dict(mae=float(np.abs(raw.astype(np.float32)-source.astype(np.float32)).mean()),timing=raw_time),
        predictions_sha256=digest(out/'predictions.json'),expert_targets_after_predictions=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    Image.fromarray(small(raw)).save(out/'raw-same-image.png')
    print(json.dumps(dict(output=str(out),summary=summary,raw_same_image=report['raw_same_image'])),flush=True)


if __name__=='__main__':main()
