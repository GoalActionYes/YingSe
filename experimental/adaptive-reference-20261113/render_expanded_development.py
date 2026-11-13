"""All development comparisons, never a selected-best image gallery."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--joint',action='store_true');args=parser.parse_args()
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v2'
    run=base/('training-expanded-joint-v1' if args.joint else 'training-expanded-control-v1')
    result=json.loads((run/'results.json').read_text(encoding='utf-8'))
    if not result['finished']:raise ValueError('Training not complete')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));manifest=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))
    if digest(data/'cohort.json')!=result['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=result['prepared_manifest_sha256']:raise ValueError('Changed development data')
    lookup={r['original_path']:r for r in manifest['files']}
    selected=json.loads((run/f'epoch-{result["best_epoch"]}/scores.json').read_text(encoding='utf-8'))
    old={r['index']:r for r in result['baseline']['cases']};new={r['index']:r for r in selected['cases']}
    out=run/'visual-review';out.mkdir(exist_ok=False);cases=[]
    def source(name):
        record=lookup[name];path=data/record['prepared_path']
        if digest(path)!=record['sha256']:raise ValueError('Changed prepared photo')
        with Image.open(path) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    def prediction(epoch,index,sha):
        path=run/f'epoch-{epoch}/{index}.png'
        if digest(path)!=sha:raise ValueError('Changed frozen prediction')
        with Image.open(path) as im:return im.convert('RGB')
    for row in cohort['groups']['validation']:
        index=row['index'];before=old[index]['mae'];after=new[index]['mae']
        canvas=Image.new('RGB',(768,575),'#222');draw=ImageDraw.Draw(canvas)
        draw.text((4,3),f'DEVELOPMENT {index}; square 256 analysis; not independent acceptance',fill='white')
        images=[('Input',source(row['content'])),('Reference',source(row['reference'])),('GT',source(row['gt'])),
            ('Current epoch8',prediction(8,index,old[index]['sha256'])),('Expanded candidate',prediction(result['best_epoch'],index,new[index]['sha256']))]
        for i,(label,im) in enumerate(images):
            x=i%3*256;y=i//3*275+20;draw.text((x+3,y+3),label,fill='white');canvas.paste(im,(x,y+19))
        severe=after-before>5 and after>before*1.25
        draw.text((516,320),f'Input MAE: {old[index]["input_mae"]:.3f}\nCurrent: {before:.3f}\nCandidate: {after:.3f}\nSevere regression: {severe}',fill='white')
        path=out/f'{index}.jpg';canvas.save(path,quality=95)
        cases.append(dict(index=index,before=before,after=after,delta=after-before,severe=severe,path=path.name,sha256=digest(path)))
    report=dict(scope='Every frozen development pair; no independent test or perceptual pass gate',script_sha256=digest(Path(__file__)),cases=cases)
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'rendered':len(cases),'severe':[r['index'] for r in cases if r['severe']]}))


if __name__=='__main__':main()
