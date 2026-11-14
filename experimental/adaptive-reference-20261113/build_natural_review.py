"""Anonymous local A/B contact sheets from previously consumed natural cases."""
import argparse,json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--current-outputs',type=Path);args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected single run name')
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1'
    manifest=json.loads((base/'results.json').read_text(encoding='utf-8'))
    out=ROOT/'outputs/adaptive-reference-20261113/natural-review'/args.run;out.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);rng=random.Random(20261115)
    current={}
    if args.current_outputs:
        current={r['index']:r for r in json.loads((args.current_outputs/'results.json').read_text())['cases']}
    cases=list(manifest['cases']);rng.shuffle(cases);key=[]
    for batch in range(8):
        canvas=Image.new('RGB',(1024,4*292),'#22252a');draw=ImageDraw.Draw(canvas)
        for position,row in enumerate(cases[batch*4:batch*4+4]):
            alias=f'C{batch*4+position+1:02d}';methods=['candidate','before'];rng.shuffle(methods)
            paths=[base/row['outputs']['input']['path'],base/'srgb'/(row['triplet']['reference']+'.png')]+[base/row['outputs'][m]['path'] for m in methods]
            if args.current_outputs:
                path=args.current_outputs/(str(row['index'])+'.png')
                if digest(path)!=current[row['index']]['sha256']:raise ValueError('Changed current product output')
                paths[2+methods.index('candidate')]=path
            for method in ['input']+methods:
                if digest(base/row['outputs'][method]['path'])!=row['outputs'][method]['sha256']:raise ValueError('Changed natural output')
            for col,(label,path) in enumerate(zip(['Source','Reference','A','B'],paths)):
                draw.text((col*256+8,position*292+8),alias+' '+label,font=font,fill='white')
                with Image.open(path) as image:
                    tile=ImageOps.contain(image.convert('RGB'),(256,256),Image.Resampling.LANCZOS)
                    canvas.paste(tile,(col*256+(256-tile.width)//2,position*292+32+(256-tile.height)//2))
            key.append(dict(alias=alias,index=row['index'],A=methods[0],B=methods[1],input_sha256=[digest(p) for p in paths]))
        canvas.save(out/f'sheet-{batch+1}.png')
    (out/'private-key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(scope='Previously consumed natural32, anonymous contact sheets; no ground-truth panel, no new model inference or independent test',dataset_sha256=digest(base/'results.json'),generator_sha256=digest(Path(__file__)),cases=32,sheets={p.name:digest(p) for p in out.glob('sheet-*.png')}),indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=32,sheets=8,scope='Anonymous review materials generated; ratings not performed')))


if __name__=='__main__':main()
