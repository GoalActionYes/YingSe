"""Counterfactual global-strength control; equal RGB displacement is not equal style."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze,algorithm_fingerprint
from reference_color_app.lut import _sample,apply_mapping


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--sweep',required=True,type=Path);args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected one run name')
    sweep=json.loads((args.sweep/'results.json').read_text());base=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1'
    cases=json.loads((base/'results.json').read_text())['cases'];lookup={r['index']:r for r in sweep['cases']}
    out=ROOT/'outputs/adaptive-reference-20261113/natural-review'/args.run;out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='Consumed natural32; choose integer0..100 old-global strength closest to current80 mean absolute RGB movement, not perceptual equivalence, quality selection or product policy',
                fixed_search=list(range(101)),tie_policy='smaller strength',global_fingerprint=algorithm_fingerprint(),sweep_sha256=digest(args.sweep/'results.json'),script_sha256=digest(Path(__file__)),cases=[],finished=False)
    for item in cases:
        target=base/item['outputs']['input']['path'];ref=base/'srgb'/(item['triplet']['reference']+'.png')
        with Image.open(target) as im:source=np.array(im.convert('RGB'))
        entry=next(row for row in lookup[item['index']]['rows'] if row['strength']==.8)['files']['adaptive'];candidate=args.sweep/entry['path']
        if digest(candidate)!=entry['sha256']:raise ValueError('Changed current80 output')
        with Image.open(candidate) as im:adaptive=np.array(im.convert('RGB'))
        desired=float(np.abs(adaptive.astype(float)-source).mean());mapping=analyze(target,ref)
        rgb=source.astype(np.float32)/np.float32(255);first=_sample(mapping.layers[0],rgb);second=_sample(mapping.layers[1],rgb)
        curve=[];best=None
        for strength in report['fixed_search']:
            t=np.float32(strength/100);area=bool(t>=np.float32(.80001));res=second if area else first
            w=np.float32(np.clip(t/np.float32(.8),0,1));w=np.float32(w*w*np.float32(3-np.float32(2)*w))
            if area:w=np.float32(1)
            mw=np.float32(np.float32(1)-t)/np.float32(.2)
            if mw>np.float32(1):mw=np.float32(0)
            res=mw*first+np.float32(np.float32(1)-mw)*res
            pixels=np.rint(np.clip(rgb*np.float32(np.float32(1)-w)+res*w,0,1)*np.float32(255)).astype(np.uint8)
            displacement=float(np.abs(pixels.astype(float)-source).mean());distance=abs(displacement-desired);curve.append(dict(strength=strength,change_mae=displacement))
            if best is None or distance<best[0]:best=(distance,strength,pixels,displacement)
        distance,strength,pixels,displacement=best;actual=apply_mapping(source,mapping,strength/100)
        assert np.array_equal(actual,pixels),'Cached interpolation differs from actual global renderer'
        path=out/f'{item["index"]}-global-matched.png';Image.fromarray(actual).save(path)
        report['cases'].append(dict(index=item['index'],selected_strength=strength,current80_change_mae=desired,global_change_mae=displacement,magnitude_gap=distance,within_half_code=distance<=.5,path=path.name,sha256=digest(path),curve=curve))
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=len(report['cases']),median_strength=float(np.median([r['selected_strength'] for r in report['cases']])),within_half_code=sum(r['within_half_code'] for r in report['cases']),maximum_gap=max(r['magnitude_gap'] for r in report['cases']))))


if __name__=='__main__':main()
