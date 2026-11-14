"""Frozen no-training neutral-anchor check on already-consumed natural32."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from neutral_anchor import correct_neutral_highlights
from reference_color_app.adjustments import transform


def blend(source,full):
    value=source.astype(np.float32)/255*np.float32(.2)+full.astype(np.float32)/255*np.float32(.8)
    return np.rint(transform(value,None)*255).astype(np.uint8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected one run name')
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1';cohort=json.loads((base/'results.json').read_text())
    sweep=ROOT/'outputs/adaptive-reference-20261113/natural-review/strength-sweep-20261115-v1';lookup={r['index']:r for r in json.loads((sweep/'results.json').read_text())['cases']}
    out=ROOT/'outputs/adaptive-reference-20261113/natural-review'/args.run;out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['neutral_anchor.py','neutral-anchor-protocol.md','probe_neutral_anchor.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed natural32 diagnostic with already-accessible GT; fixed rule, no training or product integration, no human quality gate',source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False)
    def load(path):
        with Image.open(path) as image:return np.array(image.convert('RGB'))
    for item in cohort['cases']:
        source=load(base/item['outputs']['input']['path']);ref=load(base/'srgb'/(item['triplet']['reference']+'.png'))
        row=next(r for r in lookup[item['index']]['rows'] if r['strength']==1);path=sweep/row['files']['adaptive']['path']
        if digest(path)!=row['files']['adaptive']['sha256']:raise ValueError('Changed current full output')
        full=load(path);candidate,info=correct_neutral_highlights(source,ref,full)
        before=blend(source,full);after=blend(source,candidate);truth=load(base/'srgb'/(item['triplet']['gt']+'.png'))
        truth=np.array(Image.fromarray(truth).resize((source.shape[1],source.shape[0]),Image.Resampling.BILINEAR))
        files={}
        for name,pixels in [('before80',before),('candidate80',after)]:
            path=out/f'{item["index"]}-{name}.png';Image.fromarray(pixels).save(path);files[name]=dict(path=path.name,sha256=digest(path))
        report['cases'].append(dict(index=item['index'],diagnostics=info,before_mae=float(np.abs(before.astype(float)-truth).mean()),after_mae=float(np.abs(after.astype(float)-truth).mean()),change_mae=float(np.abs(before.astype(float)-after).mean()),files=files))
    a=np.array([r['before_mae'] for r in report['cases']]);b=np.array([r['after_mae'] for r in report['cases']])
    report['summary']=dict(before_mean=float(a.mean()),after_mean=float(b.mean()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()),bypassed=sum(r['diagnostics']['bypass'] for r in report['cases']))
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['summary']))


if __name__=='__main__':main()
