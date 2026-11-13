"""Render actual saved predictions for manual segmentation inspection."""
import json
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/scene-regions/development-v1'
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'.tools/scene-segformer/config.json').read_text(encoding='utf-8'))
    colors=np.random.default_rng(938).integers(40,240,(150,3),dtype=np.uint8)
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    for i in [12,14]:
        source=data/'development-dense-round2-v1/inputs';images=[]
        for name in [f'content_709/in{i}.png',f'paired_style/tar{i}.png']:
            with Image.open(source/name) as im:
                im=im.convert('RGB');im.thumbnail((640,640),Image.Resampling.BILINEAR);images.append(np.array(im))
        panel=Image.new('RGB',(1280,520),'#222222');draw=ImageDraw.Draw(panel)
        with np.load(out/f'{i}-segmentation.npz',allow_pickle=False) as masks:
            for k,(rgb,role) in enumerate(zip(images,['target','reference'])):
                labels=masks[role+'_label'];overlay=np.rint(.6*rgb+.4*colors[labels]).astype(np.uint8)
                panel.paste(Image.fromarray(overlay),(k*640,30));draw.text((k*640+8,8),role+' predicted labels',fill='white')
                major=[int(c) for c in np.unique(labels) if (labels==c).mean()>=.015]
                for n,c in enumerate(major):
                    x=k*640+(n%4)*160;y=400+(n//4)*22
                    draw.rectangle((x,y,x+12,y+12),fill=tuple(int(v) for v in colors[c]));draw.text((x+17,y),config['id2label'][str(c)],fill='white')
        panel.save(out/f'{i}-mask-inspection.jpg',quality=95)


if __name__=='__main__':main()
