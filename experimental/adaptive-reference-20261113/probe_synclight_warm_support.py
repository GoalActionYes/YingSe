"""Screen target warmth as a spatial cue for distributing an existing grade."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from background_intervention import target_regions
from fetch_bg_real_color_directions import DEV_SHA, DOCS, read_pinned
from independent_coco import digest
from probe_bg_real_color_directions import FrozenBaseline, local_summary
from probe_bg_real_pixel_pilot import measure
from probe_gray_edge_correction import PARENT_SHA
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import mask_for, save
from probe_neutral_relation_balance import checked_image, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, read
from probe_synclight_real_pixels import LUMA, scores
from reference_color_app.media import decode_image

ROOM_SHA = '558e2b12f012c59314f30f4b0df165b565bca6ffe1ac5482ffbacac52a570afb'


def warmth_weight(target):
    rb = target[..., [0,2]].astype(np.float64)/255
    linear = np.where(rb<=.04045, rb/12.92, ((rb+.055)/1.055)**2.4)
    logratio = np.log(linear[...,0]+.01)-np.log(linear[...,1]+.01)
    sigma = .03*min(logratio.shape)
    radius = int(np.ceil(3*sigma))
    kernel = np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2)
    kernel /= kernel.sum()
    for axis in (0,1):
        pad = [(radius,radius) if i==axis else (0,0) for i in (0,1)]
        logratio = np.apply_along_axis(lambda line: np.convolve(line,kernel,mode='valid'),
                                      axis,np.pad(logratio,pad,mode='edge'))
    low,high = np.quantile(logratio,[.2,.8])
    return (2*(logratio-low)/(high-low)).clip(0,2) if high-low>1e-12 else np.ones(logratio.shape)


class WarmthProbe:
    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(same_image=True)
        weight = warmth_weight(target)
        baseline = baseline.astype(float)
        base_luma = baseline@LUMA
        neutral = base_luma[...,None]
        origin_chroma = target-(target@LUMA)[...,None]
        grade_chroma = baseline-neutral
        scaled = origin_chroma + weight[...,None]*(grade_chroma-origin_chroma)
        cap = np.ones_like(scaled)
        np.divide(255-neutral,scaled,out=cap,where=scaled>0)
        np.divide(-neutral,scaled,out=cap,where=scaled<0)
        prepared = neutral+cap.min(axis=-1).clip(0,1)[...,None]*scaled
        np.testing.assert_allclose(prepared@LUMA,base_luma,rtol=0,atol=1e-10)
        return np.rint(prepared.clip(0,255)).astype(np.uint8), dict(
            mean_weight=float(weight.mean()), fraction_at_zero=float((weight==0).mean()),
            fraction_at_two=float((weight==2).mean()))


def room_controls(out, parent, evidence, probe):
    records = {}
    for key,row in evidence['records'].items():
        folder = parent/key
        def image(name):
            path = folder/name
            assert digest(path)==row['output_sha256'][name]
            with Image.open(path) as picture:
                return np.array(picture.convert('RGB'))
        target = image('camera2-base.png')
        outputs = {}
        hashes = {}
        for label,reference,baseline in [('anchor','camera1-base.png','baseline-anchor.png'),
                                         ('cooled','camera1-cold.png','baseline-cooled.png')]:
            outputs[label],debug = probe.measure(target,image(reference),image(baseline))
            hashes[label] = dict(debug,sha256=save(out/f'{key}-{label}.png',outputs[label]))
        desired = (4*image('camera2-cold.png').astype(np.int16)+target+2)//5-target
        fraction_path = folder/'lamp-fraction.npy'
        assert digest(fraction_path)==row['fraction_sha256']
        fraction = np.load(fraction_path,allow_pickle=False)
        lightness = target@LUMA
        scopes = dict(low=fraction<=.10, high=fraction>=.30,
                      whole=np.ones(fraction.shape,bool),
                      neutral=(np.ptp(target,axis=-1)<=20)&(lightness>=40)&(lightness<=220))
        regions = scores(outputs['cooled'].astype(np.int16)-outputs['anchor'],desired,scopes)
        passed = row['local_eligible'] and regions['low']['mae']<=1 and regions['high']['passed']
        records[key] = dict(regions=regions,passed=passed,output=hashes)
    return records


def photo_controls(out, held, parent, probe):
    report = {}
    for key,row in held['cases'].items():
        pair = row['target_pair']
        case = parent[pair]
        target = read(case['input_file'])
        outputs, baseline, hashes = {}, {}, {}
        for role in ('C1','F1'):
            filename = f'{key}-{role}.png'
            source = row['input'][role]
            path = BASE/'bg-real-pixel-inputs'/source['name']
            assert digest(path)==source['sha256']
            baseline[role] = checked_image(BASE/'bg-real-pixel-pilot'/filename,row['outputs'][role])
            outputs[role],debug = probe.measure(target,decode_image(path),baseline[role])
            hashes[role] = dict(debug,sha256=save(out/filename,outputs[role]))
        regions = target_regions(target,mask_for(case,pair.split('-')[0]))
        neutral = regions['neutral_proxy']
        values = {name:[measure(item['C1'],item['F1'],mask)['mae'] for item in (baseline,outputs)]
                  for name,mask in [('neutral',neutral),('neutral_person',neutral&regions['person'])]}
        safe = all(after<=before+1 for before,after in values.values())
        report[key] = dict(target_pair=pair, values=values, safe=safe, output=hashes)
    return report


def run(out):
    start = time.monotonic()
    room = read_pinned(DOCS/'2026-11-18-synclight-ambient-balance-evidence.json',ROOM_SHA)
    held = read_pinned(DOCS/'2026-11-18-bg-real-pixel-pilot-evidence.json',DEV_SHA)
    toggle = read_pinned(DOCS/'2026-11-18-local-edit-toggle-evidence.json',PARENT_REPORT)
    parent = {row['pair']:row for row in read_pinned(PARENT/'results.json',PARENT_SHA)['cases']}
    probe = WarmthProbe()
    sample = np.full((16,16,3),140,np.uint8)
    np.testing.assert_array_equal(probe.measure(sample,sample.copy(),255-sample)[0],sample)
    out.mkdir()
    rooms = room_controls(out,BASE/'synclight-ambient-balance',room,probe)
    photos = photo_controls(out,held,parent,probe)
    local = local_summary(local_controls(probe,out,toggle,parent))
    baseline_dir = out/'local-baseline'
    baseline_dir.mkdir()
    local_base = local_summary(local_controls(FrozenBaseline(),baseline_dir,toggle,parent))
    gates = dict(rooms=all(row['passed'] for row in rooms.values()),
                 room_neutral=rooms['scene25']['regions']['neutral']['mae']
                 <= room['records']['scene25']['scores']['neutral']['mae']+.25,
                 portraits=all(row['safe'] for row in photos.values()),
                 whole=all(not .75<=local_base[key]['whole_projection']<=1.25 or .75<=row['whole_projection']<=1.25
                           for key,row in local.items()),
                 background=all(row['local_background_projection']>=local_base[key]['local_background_projection']-.10
                                for key,row in local.items() if row['gates']['positive']),
                 empty=all(row['gates']['checks']['no_person'] is not False for row in local.values()))
    assert time.monotonic()-start<=120
    source = Path(__file__)
    result = dict(rooms=rooms,photos=photos,local=local,local_base=local_base,gates=gates,
                  proceed=all(gates.values()),room_sha256=ROOM_SHA,photo_sha256=DEV_SHA,toggle_sha256=PARENT_REPORT,
                  source_sha256=digest(source),protocol_sha256=digest(source.with_name('synclight-warm-support-protocol.md')))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(gates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
