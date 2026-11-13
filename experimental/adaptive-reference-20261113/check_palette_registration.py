"""Frozen controlled and photographic development comparison."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest
from palette_registration import palette,register,apply


def load(p):
    with Image.open(p) as im:return np.array(im.convert('RGB'))


def small(rgb):
    im=Image.fromarray(rgb);im.thumbnail((640,640),Image.Resampling.BILINEAR);return np.array(im)


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/palette-registration/development-v1';out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['palette_registration.py','check_palette_registration.py','palette-registration-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    data=ROOT/'outputs/adaptive-reference-20261113/pst50';prior=data/'development-canonical-final-v1'
    controls=[];control=ROOT/'outputs/adaptive-reference-20261113/composition/development-v1'
    for look in ['identity','warm']:
        for n in range(4):
            name=f'control-{look}-{n}';target=load(control/(name+'-target.png'));reference=load(control/(name+'-reference.png'))
            matrix,diagnostics=register(palette(target),palette(reference));rgb=apply(target,matrix)
            expected=target if look=='identity' else np.rint(np.clip(target.astype(float)*[1.05,.98,.9]+[2.55,1.275,5.1],0,255)).astype(np.uint8)
            Image.fromarray(rgb).save(out/(name+'.png'));controls.append(dict(look=look,variant=n,mae=float(np.abs(rgb.astype(float)-expected).mean()),matrix=matrix.tolist(),diagnostics=diagnostics))
    rows=[];report=dict(scope='Consumed development only, not semantic or product acceptance',controls=controls,cases=rows,source_sha256={p.name:digest(p) for p in frozen.iterdir()})
    for i in range(1,16):
        inputs=data if i<=5 else data/'development-dense-round2-v1/inputs';a,b=inputs/f'content_709/in{i}.png',inputs/f'paired_style/tar{i}.png'
        row=dict(index=i,source_sha256=dict(target=digest(a),reference=digest(b)));rows.append(row)
        try:
            target,reference=load(a),load(b);start=time.perf_counter();matrix,diag=register(palette(target),palette(reference));rgb=apply(target,matrix)
            row.update(matrix=matrix.tolist(),diagnostics=diag,processing_seconds=time.perf_counter()-start)
            p=out/f'{i}-output.png';Image.fromarray(small(rgb)).save(p);row['sha256']=digest(p)
        except Exception as exc:row['failure']=dict(type=type(exc).__name__,message=str(exc))
        print(json.dumps(dict(index=i,failure=row.get('failure'))),flush=True)
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for row in rows:
        if 'failure' in row:continue
        i=row['index'];inputs=data if i<=5 else data/'development-dense-round2-v1/inputs'
        if digest(out/f'{i}-output.png')!=row['sha256']:raise ValueError('Prediction changed')
        truth=small(load(inputs/f'paired_gt/gt{i}.png'));candidate=load(out/f'{i}-output.png');current=load(prior/f'{i}-current.png')
        row['scores']={k:float(np.abs(v.astype(np.float32)-truth).mean()) for k,v in [('current',current),('registration',candidate)]}
        panel=Image.new('RGB',(1080,520),'#222222');draw=ImageDraw.Draw(panel)
        for n,(label,rgb) in enumerate([('Input',small(load(inputs/f'content_709/in{i}.png'))),('Reference',small(load(inputs/f'paired_style/tar{i}.png'))),('Expert',truth),('Current',current),('Palette registration',candidate)]):
            im=Image.fromarray(rgb);im.thumbnail((356,230));x,y=n%3*360,n//3*260;panel.paste(im,(x,y+25));draw.text((x+8,y+5),label,fill='white')
        panel.save(out/f'{i}-comparison.jpg',quality=94)
    valid=[r for r in rows if 'scores' in r];base=np.array([r['scores']['current'] for r in valid]);actual=np.array([r['scores']['registration'] for r in valid])
    report.update(summary=dict(total=15,failed=15-len(valid),current_mean=float(base.mean()),candidate_mean=float(actual.mean()),improved=int((actual<base-.1).sum()),regressed=int((actual>base+.1).sum()),severe=int(((actual>base+5)&(actual>base*1.25)).sum())),predictions_sha256=digest(out/'predictions.json'),all_predictions_before_expert_read=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=report['summary'],control_maes=[r['mae'] for r in controls])),flush=True)


if __name__=='__main__':main()
