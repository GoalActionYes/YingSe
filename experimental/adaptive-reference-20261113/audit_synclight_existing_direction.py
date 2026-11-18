"""Bound what spatial amplification of the already-rendered 13 change can express."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from audit_bg_real_skin_regions import BASE, ROOT, digest

BASELINE_SHA = '558e2b12f012c59314f30f4b0df165b565bca6ffe1ac5482ffbacac52a570afb'
REC709 = np.array([2126, 7152, 722])


def cube_bound(anchor, direction):
    fraction = np.full(direction.shape, 2., dtype=np.float64)
    np.divide(255-anchor, direction, out=fraction, where=direction>0)
    np.divide(-anchor, direction, out=fraction, where=direction<0)
    return np.minimum(2, fraction.min(axis=-1))


def optimum(direction, expected, upper):
    ratios = np.zeros_like(direction, dtype=np.float64)
    np.divide(expected, direction, out=ratios, where=direction!=0)
    choices = np.concatenate((np.zeros_like(upper[...,None]), upper[...,None],
                              ratios.clip(0,upper[...,None])),axis=-1)
    attained = choices[...,None]*direction[...,None,:]
    costs = np.abs(attained-expected[...,None,:]).sum(axis=-1)
    selection = np.argmin(costs,axis=-1)
    weight = np.take_along_axis(choices,selection[...,None],axis=-1)[...,0]
    return weight, np.min(costs,axis=-1)


def audit_scene(parent, evidence, out, key):
    def pixels(name):
        path = parent/name
        assert digest(path)==evidence['output_sha256'][name]
        with Image.open(path) as image:
            return np.array(image.convert('RGB'),dtype=np.int16)
    anchor = pixels('baseline-anchor.png')
    cooled = pixels('baseline-cooled.png')
    base = pixels('camera2-base.png')
    cold = pixels('camera2-cold.png')
    on_disk = parent/'lamp-fraction.npy'
    assert digest(on_disk)==evidence['fraction_sha256']
    fraction=np.load(on_disk,allow_pickle=False)
    expected = (4*cold+base+2)//5-base
    direction = cooled-anchor
    limit = cube_bound(anchor,direction)
    weight, cost = optimum(direction,expected,limit)
    raw = anchor+weight[...,None]*direction
    np.testing.assert_array_less(raw.max(),255+1e-10)
    np.testing.assert_array_less(-raw.min(),1e-10)
    image = np.rint(raw.clip(0,255)).astype(np.uint8)
    output = out/f'{key}-best-l1.png'
    Image.fromarray(image).save(output)
    rgb = image.astype(np.int16)-anchor
    luma = base@REC709
    masks = dict(whole=np.ones(fraction.shape,bool),low=fraction<=.10,high=fraction>=.30,
                 neutral=(np.ptp(base,axis=-1)<=20)&(luma>=400000)&(luma<=2200000))
    records={}
    for name,area in masks.items():
        n=int(area.sum())
        if n<256:
            records[name]=dict(pixels=n,qualified=False)
            continue
        expected_local = expected[area].astype(np.float64)
        direction_local = direction[area].astype(np.float64)
        energy = float(np.square(expected_local).sum())
        assert energy>0 and n==evidence['scores'][name]['pixels']
        dots = (expected_local*direction_local).sum(axis=-1)
        continuous_max = float((np.maximum(dots,0)*limit[area]).sum()/energy)
        rounding = float(.5*np.abs(expected_local).sum()/energy)
        minimum = float(cost[area].mean()/3)
        threshold = 1. if name=='low' else .85*float(np.abs(expected_local).mean())
        floor = max(0.,minimum-.5)
        ceiling = continuous_max+rounding
        witness = rgb[area].astype(np.float64)
        mae = float(np.abs(witness-expected_local).mean())
        projection = float((witness*expected_local).sum()/energy)
        records[name]=dict(pixels=n,qualified=True,minimum_float_mae=minimum,quantized_mae_lower_bound=floor,
                           float_direction_ceiling=continuous_max,quantized_direction_upper_bound=ceiling,
                           witness_mae=mae,witness_projection=projection,error_limit=threshold,
                           error_impossible=floor>threshold,
                           direction_impossible=ceiling<.75 if name=='high' else None,
                           witness_pass=mae<=threshold and (name!='high' or .75<=projection<=1.25))
    weak,strong=(records[region] for region in ('low','high'))
    blocked=weak['error_impossible'] or strong['error_impossible'] or strong['direction_impossible']
    return dict(regions=records, ruled_out=blocked, witness_pass=weak['witness_pass'] and strong['witness_pass'],
                output_sha256=digest(output))


def run(out):
    start=time.monotonic()
    evidence=ROOT/'docs/research/2026-11-18-synclight-ambient-balance-evidence.json'
    assert digest(evidence)==BASELINE_SHA
    data=json.loads(evidence.read_bytes())
    out.mkdir()
    parent=BASE/'synclight-ambient-balance'
    records={name:audit_scene(parent/name,row,out,name) for name,row in data['records'].items()}
    assert time.monotonic()-start<=60
    source=Path(__file__)
    report=dict(records=records,baseline_sha256=BASELINE_SHA,source_sha256=digest(source),
                protocol_sha256=digest(source.with_name('synclight-existing-direction-bound-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({name:(row['ruled_out'],row['witness_pass'],row['regions']['high']) for name,row in records.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
