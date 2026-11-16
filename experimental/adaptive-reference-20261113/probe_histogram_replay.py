"""Replay fixed v2 images through separated reference conditions, no new model."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine,PreparedSpatial,finish_spatial
from background_intervention import target_regions,region_delta
from histogram_replay import conditions


def main():
    torch.set_num_threads(4)
    base=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
    data=json.loads((base/'results.json').read_text());assert data['finished'] and len(data['cases'])==6
    out=ROOT/'outputs/adaptive-reference-20261113/histogram-replay-20261116-v1';out.mkdir(exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for n in ['histogram-replay-protocol.md','histogram_replay.py','test_histogram_replay.py','probe_histogram_replay.py','spatial_render.py','frame_transfer.py','background_intervention.py']:
        (frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    engine=SpatialEngine('multiref-v1');assert engine.checkpoint_sha==data['model_sha256']
    report=dict(scope='Fixed consumed v2 references, Y/CbCr mechanism replay; not quality evaluation',
        input_results_sha256=digest(base/'results.json'),model_sha256=engine.checkpoint_sha,
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))

    def read(entry):
        p=base/entry['path'];assert digest(p)==entry['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB'))

    def save(p,rgb):
        Image.fromarray(rgb).save(p);return dict(path=str(p.relative_to(out)),sha256=digest(p))

    with torch.inference_mode():
        for case in data['cases']:
            folder=out/case['pair'];folder.mkdir();target=read(case['input_file'])
            with Image.open(base/case['pair']/'target-mask.png') as im:mask=np.array(im)>0
            regions=target_regions(target,mask)
            c=engine.analysis(target);hc=engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](c))
            low=engine.api['TF'].resize(c,(64,64),antialias=True)
            histogram=lambda ref:engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](engine.analysis(ref))).cpu().numpy()
            h0=histogram(read(case['variants'][0]['reference']))
            def infer(h):
                hist=torch.cat([hc,torch.from_numpy(h).to(engine.device)],1)
                grid=engine.api['lowpass'](engine.model.apply_grid(low,hist))
                assert torch.isfinite(grid).all()
                pixels=finish_spatial(target,PreparedSpatial(engine,grid,{}).apply(target,1),.8)
                return pixels,grid.detach().cpu().numpy(),engine.model.hist_projection(hist).detach().cpu().numpy()
            baseline,g0,p0=infer(h0)
            assert np.array_equal(baseline,read(case['variants'][0]['methods']['adaptive']['file']))
            row=dict(pair=case['pair'],group=case['group'],baseline=save(folder/'baseline.png',baseline),variants=[])
            for v in case['variants']:
                if not v['name'].startswith('background'):continue
                assert not v['certified_aligned']
                changed=histogram(read(v['reference']));variants=conditions(h0,changed)
                archive=folder/(v['name']+'-conditions.npz');np.savez_compressed(archive,h0=h0,h1=changed,**variants)
                record=dict(name=v['name'],conditions_file=dict(path=str(archive.relative_to(out)),sha256=digest(archive)),
                    histogram_l1_by_channel=np.abs(changed-h0).sum((0,2)).tolist(),methods={})
                for name,h in variants.items():
                    pixels,grid,projection=infer(h)
                    if name=='full':assert np.array_equal(pixels,read(v['methods']['adaptive']['file']))
                    record['methods'][name]=dict(file=save(folder/(v['name']+'-'+name+'.png'),pixels),
                        grid_pixel_sha256=__import__('hashlib').sha256(grid.tobytes()).hexdigest(),
                        grid_mae=float(np.abs(grid-g0).mean()),projection_mae=float(np.abs(projection-p0).mean()),
                        regions={n:region_delta(pixels,baseline,m) for n,m in regions.items()})
                row['variants'].append(record)
            report['cases'].append(row)
            (out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(json.dumps(dict(pair=case['pair'],completed=len(report['cases']))),flush=True)
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    assert digest(base/'results.json')==report['input_results_sha256']
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(finished=True,outputs=114,product_replays=42)),flush=True)


if __name__=='__main__':main()
