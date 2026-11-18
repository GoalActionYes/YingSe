"""Create a scene-balanced weak/strong illumination contrast from fixed captures."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from independent_coco import ROOT, digest
from probe_synclight_real_pixels import GAIN, LUMA, save, scores, shot, srgb8
from spatial_render import SpatialEngine, finish_spatial


def fraction(ambient, lamp, ambient_gain):
    light = lamp@LUMA
    combined = ambient_gain*(ambient@LUMA)+light
    return np.divide(light, combined, out=np.zeros_like(light), where=combined>0), combined


def reconstruct_controls(components, multiplier):
    radiance = [(multiplier*ambient+lamp, multiplier*ambient+lamp*GAIN)
                for ambient, lamp in components]
    exposure = .70/max(float(np.quantile(base@LUMA,.99)) for base,_ in radiance)
    pairs = [(srgb8(base*exposure), srgb8(cold*exposure)) for base,cold in radiance]
    return pairs, exposure


def run(decoder, sources, parent, out):
    started = time.monotonic()
    sys.path.insert(0, str(decoder.resolve()))
    import rawpy

    torch.set_num_threads(4)
    evidence_path = ROOT/'docs/research/2026-11-18-synclight-real-pixel-evidence.json'
    evidence = json.loads(evidence_path.read_bytes())
    intake = json.loads((sources/'intake.json').read_bytes())
    assert digest(sources/'intake.json') == evidence['intake_sha256']
    assert rawpy.__version__ == evidence['rawpy']
    assert all(digest(sources/name)==row['sha256'] for name,row in intake['files'].items())
    original_fractions = {}
    for scene, row in evidence['records'].items():
        path = parent/scene/'lamp-fraction.npy'
        assert row['qualified'] and digest(path)==row['fraction_sha256']
        original_fractions[scene] = np.load(path,allow_pickle=False)
    quantiles = {name:float(np.quantile(plane,.05,method='higher'))
                 for name,plane in original_fractions.items()}
    upper = max(quantiles.values())
    assert 0 < upper < 1
    multiplier = math.ceil(9*upper/(1-upper))

    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha==evidence['weight_sha256']
    out.mkdir()
    records = {}
    for scene, row in evidence['records'].items():
        pairs = []
        folder = out/scene
        folder.mkdir()
        for view in (1,2):
            channels = [shot(rawpy, sources/f'{scene}_{view}_{role}.DNG', row['white_balance'])[0]
                        for role in ('0_ambient','1_camvis')]
            pairs.append(channels)
        np.testing.assert_allclose(fraction(*pairs[1],1)[0], original_fractions[scene], rtol=0, atol=1e-12)
        controls, exposure = reconstruct_controls(pairs,multiplier)
        (base_reference,cold_reference),(target,target_cold) = controls
        output_hashes = {}
        for view,(base,cold) in enumerate(controls,1):
            for role,image in [('base',base),('cold',cold)]:
                name = f'camera{view}-{role}.png'
                output_hashes[name] = save(folder/name,image)
        anchor,cooled = [finish_spatial(target,engine.prepare(target,reference).apply(target),.8)
                         for reference in (base_reference,cold_reference)]
        for name,image in [('baseline-anchor.png',anchor),('baseline-cooled.png',cooled)]:
            output_hashes[name] = save(folder/name,image)
        power, _ = fraction(*pairs[1],multiplier)
        np.save(folder/'lamp-fraction.npy',power,allow_pickle=False)
        expected = finish_spatial(target,target_cold,.8).astype(np.int16)-target
        actual = cooled.astype(np.int16)-anchor
        intensity = target@LUMA
        masks = dict(whole=np.ones(power.shape,bool), low=power<=.10, high=power>=.30,
                     neutral=(np.ptp(target,axis=-1)<=20)&(intensity>=40)&(intensity<=220))
        regional = scores(actual,expected,masks)
        low,high = (regional[name] for name in ('low','high'))
        coverage = {name:regional[name]['pixels']/power.size for name in ('low','high')}
        local_eligible = min(coverage.values())>=.05 and high.get('no_transfer',0)>=2
        weak_error = float(np.abs((actual-expected)[masks['low']]).mean()) if low['pixels'] else None
        passes = local_eligible and weak_error<=1 and high['passed']
        records[scene] = dict(exposure=exposure, coverage=coverage, scores=regional, weak_error=weak_error,
                              local_eligible=local_eligible, local_pass=passes, output_sha256=output_hashes,
                              fraction_sha256=digest(folder/'lamp-fraction.npy'))
    source = Path(__file__)
    assert time.monotonic()-started<=120
    result = dict(records=records, target_quantiles=quantiles, ambient_multiplier=multiplier,
                  parent_sha256=digest(evidence_path), source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('synclight-ambient-balance-protocol.md')))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({name:(row['coverage'],row['scores']['high'],row['weak_error'],row['local_pass'])
           for name,row in records.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoder','sources','parent','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    run(args.decoder,args.sources,args.parent,args.out)
