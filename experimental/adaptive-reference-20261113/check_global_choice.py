"""One fixed extension on fifteen already-used development examples, not a new test."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from canonical_model import ROOT
from independent_coco import digest
from dense_anchor_model import DenseAnchorModel
from dense_anchors import match_anchors
from global_choice import choose_global
from check_dense_anchors import load,tensor,rgb8


def main():
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    out=data/'development-global-choice-v1'; out.mkdir(exist_ok=False)
    frozen=out/'frozen-source'; frozen.mkdir(); hashes={}
    for name in ('global_choice.py','check_global_choice.py','global-choice-protocol.md','dense_anchors.py','dense_anchor_model.py','check_dense_anchors.py'):
        source=Path(__file__).with_name(name); (frozen/name).write_bytes(source.read_bytes()); hashes[name]=digest(source)
    torch.set_num_threads(4); model=DenseAnchorModel(); cases=[]
    report=dict(scope='Fifteen used development pairs; not heldout or product acceptance',source_sha256=hashes,cases=cases)
    for index in range(1,16):
        source=data if index<=5 else data/'development-dense-round2-v1/inputs'
        target=load(source/f'content_709/in{index}.png'); reference=load(source/f'paired_style/tar{index}.png')
        prediction_source=data/('development-dense-anchors-native-v1' if index<=5 else 'development-dense-round2-v1/predictions')
        base=load(prediction_source/f'{index}-current.png'); canonical=load(prediction_source/f'{index}-canonical.png')
        features,_,grid,_=model.extract(target); ref_features,ref_colors,_,_=model.extract(reference)
        with torch.inference_mode():
            anchors,confidence=match_anchors(features,ref_features,ref_colors)
            output,diagnostics=choose_global([tensor(x,model.device) for x in (target,base,canonical)],
                anchors.T.reshape(1,3,*grid),confidence.reshape(1,1,*grid))
        rgb=rgb8(output); path=out/f'{index}-chosen.png'; Image.fromarray(rgb).save(path)
        cases.append(dict(index=index,diagnostics=diagnostics,path=path.name,sha256=digest(path),
            input_sha256=dict(target=digest(source/f'content_709/in{index}.png'),reference=digest(source/f'paired_style/tar{index}.png'))))
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Freeze all outputs before any read of the expert targets in this invocation.
    for row in cases:
        index=row['index']; source=data if index<=5 else data/'development-dense-round2-v1/inputs'
        prediction_source=data/('development-dense-anchors-native-v1' if index<=5 else 'development-dense-round2-v1/predictions')
        truth=load(source/f'paired_gt/gt{index}.png')
        images=dict(input=load(source/f'content_709/in{index}.png'),current=load(prediction_source/f'{index}-current.png'),
            canonical=load(prediction_source/f'{index}-canonical.png'),chosen=load(out/row['path']))
        row['scores']={name:float(np.abs(rgb.astype(np.float32)-truth.astype(np.float32)).mean()) for name,rgb in images.items()}
        row['output_vs_input_mae']=float(np.abs(images['chosen'].astype(np.float32)-images['input'].astype(np.float32)).mean())
        panel=Image.new('RGB',(1080,520),'#222222'); draw=ImageDraw.Draw(panel)
        for k,(name,rgb) in enumerate([('Input',images['input']),('Reference',load(source/f'paired_style/tar{index}.png')),('Expert target',truth),
            ('Current',images['current']),('Canon',images['canonical']),('Chosen',images['chosen'])]):
            im=Image.fromarray(rgb); im.thumbnail((356,232)); x,y=k%3*360,k//3*260
            panel.paste(im,(x+(360-im.width)//2,y+24+(232-im.height)//2)); draw.text((x+8,y+6),name,fill='white')
        panel.save(out/f'{index}-comparison.jpg',quality=94)
    summary={}
    for name in ('input','current','canonical','chosen'):
        a=np.array([r['scores'][name] for r in cases]); b=np.array([r['scores']['current'] for r in cases])
        summary[name]=dict(mean=float(a.mean()),median=float(np.median(a)),improved=int((a<b-.1).sum()),regressed=int((a>b+.1).sum()),
            severe=int(((a>b+5)&(a>b*1.25)).sum()))
    report.update(summary=summary,predictions_sha256=digest(out/'predictions.json'),expert_targets_read_after_predictions=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),summary=summary,weights=[dict(index=r['index'],weights=r['diagnostics']['weights']) for r in cases])),flush=True)


if __name__=='__main__': main()
