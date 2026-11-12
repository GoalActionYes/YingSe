import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
from controlled_styles import stylize,PROFILES
from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=('development','evaluation'),default='development')
    args=parser.parse_args()
    data=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    out=data/args.split
    (out/'references').mkdir(parents=True,exist_ok=True);(out/'truth').mkdir(exist_ok=True)
    jobs=[]
    pairs=[p for p in manifest['pairs'] if p['split']==args.split]
    sheet=Image.new('RGB',(4*240,3*240),'#202020');draw=ImageDraw.Draw(sheet)
    for row,pair in enumerate(pairs):
        target=np.array(Image.open(data/f"{pair['target']}-input.png"))
        reference=np.array(Image.open(data/f"{pair['reference']}-input.png"))
        masks={role:np.asarray(Image.open(data/f"{pair[role]}-human-mask.png").filter(ImageFilter.GaussianBlur(2)),dtype=float)/255 for role in ('target','reference')}
        for profile in PROFILES:
            refpath=out/'references'/f"{pair['reference']}-{profile}.png"
            truthpath=out/'truth'/f"{pair['target']}-{profile}.png"
            Image.fromarray(stylize(reference,masks['reference'],profile)).save(refpath)
            Image.fromarray(stylize(target,masks['target'],profile)).save(truthpath)
            jobs.append(dict(**pair,profile=profile,reference_file=str(refpath.relative_to(data)),
                             truth_file=str(truthpath.relative_to(data))))
        if args.split=='development':
            for col,array in enumerate((target,reference)):
                picture=Image.fromarray(array);picture.thumbnail((240,210),Image.Resampling.BILINEAR)
                x=(row%2)*480+col*240;y=(row//2)*240
                sheet.paste(picture,(x,y+25));draw.text((x+3,y+4),pair['group']+' / '+('target' if col==0 else 'reference'),fill='white')
    (out/'jobs.json').write_text(json.dumps(dict(split=args.split,jobs=jobs,
         controlled_style_source_sha256=digest(Path(__file__).with_name('controlled_styles.py'))),indent=2),encoding='utf-8')
    if args.split=='development':sheet.save(out/'input-pairs.jpg')
    print(json.dumps(dict(split=args.split,pairs=len(pairs),cases=len(jobs))))


if __name__=='__main__':main()
