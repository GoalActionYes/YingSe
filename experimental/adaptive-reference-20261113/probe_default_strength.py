"""Actual product strength sweep on consumed natural32; no model/default changes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from spatial_store import SpatialStore
from spatial_render import SpatialEngine,finish_spatial
from reference_color_app.pipeline import analyze,algorithm_fingerprint
from reference_color_app.lut import apply_mapping


def describe(rgb,source):
    x=rgb.astype(float);y=x@[.299,.587,.114]
    return dict(source_change_mae=float(np.abs(x-source).mean()),brightness=float(y.mean()),contrast=float(y.std()),
                saturation_range=float((x.max(-1)-x.min(-1)).mean()),endpoint_fraction=float(((rgb==0)|(rgb==255)).mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected one run name')
    out=ROOT/'outputs/adaptive-reference-20261113/natural-review'/args.run;out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_default_strength.py','spatial_store.py','spatial_render.py','aligned_spatial.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1';manifest=json.loads((base/'results.json').read_text())
    if len(manifest['cases'])!=32:raise ValueError('Expected fixed natural32')
    engine=SpatialEngine('multiref-v1');store=SpatialStore(out/'data',engine)
    report=dict(scope='Consumed natural32 at0/50/80/100; actual product paths, not naturalness or human preference scores',strengths=[0,.5,.8,1],model_sha256=engine.checkpoint_sha,
                source_sha256={p.name:digest(p) for p in frozen.iterdir()},global_fingerprint=algorithm_fingerprint(),dataset_sha256=digest(base/'results.json'),cases=[],finished=False)
    for item in manifest['cases']:
        target=base/item['outputs']['input']['path'];ref=base/'srgb'/(item['triplet']['reference']+'.png')
        if digest(target)!=item['outputs']['input']['sha256']:raise ValueError('Changed natural source')
        a,b=store.add_media(target),store.add_media(ref);m=store.analyze(a['id'],b['id']);source,_=store.decode(a)
        baseline=analyze(target,ref);rows=[];full=store.render(m['id'],source,1)
        for strength in report['strengths']:
            adaptive=store.render(m['id'],source,strength);global_rgb=apply_mapping(source,baseline,strength)
            assert np.array_equal(adaptive,finish_spatial(source,full,strength))
            if strength==0:assert np.array_equal(adaptive,source) and np.array_equal(global_rgb,source)
            files={}
            for method,pixels in [('adaptive',adaptive),('global',global_rgb)]:
                path=out/f'{item["index"]}-{method}-{round(strength*100)}.png';Image.fromarray(pixels).save(path);files[method]=dict(path=path.name,sha256=digest(path))
            naive=finish_spatial(source,apply_mapping(source,baseline,1),strength)
            rows.append(dict(strength=strength,files=files,adaptive=describe(adaptive,source),global_=describe(global_rgb,source),
                             global_vs_naive_full_blend_mae=float(np.abs(global_rgb.astype(float)-naive).mean())))
        report['cases'].append(dict(index=item['index'],source_sha256=digest(target),reference_sha256=digest(ref),certified_aligned=m['diagnostics']['certified_aligned']['accepted'],rows=rows))
        if len(report['cases'])%8==0:print(json.dumps(dict(cases=len(report['cases']))),flush=True)
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,cases=32,images=256)))


if __name__=='__main__':main()
