"""All eight fixed author sample pairs; no natural-image ground truth score."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from canonical_model import CanonicalModel
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/author-samples'
    out.mkdir(parents=True,exist_ok=True)
    upstream=ROOT/'.tools/CanonCGT/samples'
    engine=CanonicalModel()
    rows=[]
    for path in sorted((upstream/'inp').glob('*.png')):
        name=path.name
        arrays=[]
        for folder in ('inp','ref'):
            image=Image.open(upstream/folder/name).convert('RGB')
            image.thumbnail((640,640),Image.Resampling.BILINEAR)
            arrays.append(np.array(image))
        target,reference=arrays
        targetpath=out/f'{path.stem}-target.png'; referencepath=out/f'{path.stem}-reference.png'
        Image.fromarray(target).save(targetpath);Image.fromarray(reference).save(referencepath)
        result,timing=engine.predict(target,reference)
        baseline=apply_mapping(target,analyze(targetpath,referencepath),1.)
        panel=Image.new('RGB',(1280,352),'#202020'); draw=ImageDraw.Draw(panel)
        for k,(label,array) in enumerate(zip(('target','reference','global','author E2E model'),[target,reference,baseline,result])):
            image=Image.fromarray(array);image.thumbnail((320,320),Image.Resampling.BILINEAR)
            panel.paste(image,(k*320+(320-image.width)//2,32));draw.text((k*320+5,8),label,fill='white')
        panel.save(out/f'{path.stem}-comparison.png')
        Image.fromarray(result).save(out/f'{path.stem}-model.png')
        rows.append(dict(file=name,**timing))
    report=dict(scope='Eight author-selected demonstration pairs; not an independent evaluation or ground truth',
                provenance=engine.provenance,cases=rows)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
