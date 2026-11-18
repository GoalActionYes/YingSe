"""Render fixed physical-lighting controls and check the current two-image grade."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from independent_coco import ROOT, digest
from spatial_render import SpatialEngine, finish_spatial


LUMA = np.array([.2126, .7152, .0722])
GAIN = np.array([.6, 1., 1.4])


def source_camera(raw):
    kind = 'rgb_xyz_matrix' if raw.rgb_xyz_matrix.any() else 'color_matrix'
    matrix = getattr(raw, kind)
    if not np.isfinite(matrix).all() or not matrix.any():
        raise ValueError('Missing camera calibration')
    return dict(matrix=matrix.tolist(), matrix_field=kind, shape=list(raw.raw_image_visible.shape),
                black=raw.black_level_per_channel, white=raw.white_level,
                pattern=None if raw.raw_pattern is None else raw.raw_pattern.tolist(),
                room_wb=list(raw.camera_whitebalance), flip=raw.sizes.flip)


def thumbnail(rgb):
    height, width = rgb.shape[:2]
    size = tuple(round(512*edge/max(height, width)) for edge in (width,height))
    planes = [np.asarray(Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32))
                         .resize(size, Image.Resampling.BOX), dtype=np.float64)
              for channel in np.moveaxis(rgb, -1, 0)]
    return np.stack(planes, axis=-1)/65535


def shot(rawpy, path, wb):
    with Image.open(path) as image:
        tags = image.getexif()
        capture = tags.get_ifd(34665)
        metadata = [tags.get(272), *(str(capture[key]) for key in (33434,33437,34855))]
    with rawpy.imread(str(path)) as raw:
        identity = source_camera(raw)
        rgb = raw.postprocess(user_wb=wb, use_auto_wb=False, use_camera_wb=False, gamma=(1,1),
                              no_auto_bright=True, output_color=rawpy.ColorSpace.sRGB,
                              output_bps=16, half_size=True)
    identity['capture'] = metadata
    return thumbnail(rgb), identity, float((rgb == 65535).mean())


def registration(ambient, lamp):
    first, second = (np.hypot(*np.gradient(np.log1p((value @ LUMA)*64))) for value in (ambient, lamp))
    height, width = first.shape
    target = first[4:-4,4:-4].ravel()
    stats = {}
    for y in range(-4,5):
        for x in range(-4,5):
            patch = second[4+y:height-4+y,4+x:width-4+x].ravel()
            stats[x,y] = float(np.corrcoef(target, patch)[0,1])
    assert all(np.isfinite(score) for score in stats.values())
    best = max(stats, key=stats.get)
    return dict(shift=list(best), zero=stats[0,0], best=stats[best],
                qualified=stats[0,0]>=.50 and max(map(abs,best))<=1)


def srgb8(linear):
    values = linear.clip(0,1)
    return np.rint(np.where(values<=.0031308,12.92*values,1.055*values**(1/2.4)-.055)*255).astype(np.uint8)


def save(path, image):
    Image.fromarray(image).save(path)
    return digest(path)


def scores(actual, expected, masks):
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count < 256:
            result[name] = dict(pixels=count, qualified=False)
            continue
        a, b = (np.asarray(array[mask], dtype=np.float64) for array in (actual, expected))
        power = float((b*b).sum())
        signal = float(np.abs(b).mean())
        error = float(np.abs(a-b).mean())
        alignment = float((a*b).sum()/power) if power else None
        result[name] = dict(pixels=count, qualified=power>0, direction=alignment, mae=error, no_transfer=signal,
                            passed=bool(alignment is not None and .75<=alignment<=1.25 and error<=.85*signal))
    return result


def run(decoder, sources, out):
    start = time.monotonic()
    sys.path.insert(0, str(decoder.resolve()))
    import rawpy

    assert rawpy.__version__ == '0.28.0'
    intake_path = sources/'intake.json'
    intake = json.loads(intake_path.read_bytes())
    evidence = ROOT/'docs/research/2026-11-18-synclight-real-intake-evidence.json'
    assert intake['source_sha256']==digest(evidence)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    out.mkdir()
    records = {}
    for scene in ('scene35','scene25'):
        folder = out/scene
        folder.mkdir()
        paths = [{role:sources/f'{scene}_{view}_{role}.DNG' for role in ('0_ambient','1_camvis')} for view in (1,2)]
        for view in paths:
            for path in view.values():
                assert digest(path)==intake['files'][path.name]['sha256']
        with rawpy.imread(str(paths[0]['0_ambient'])) as raw:
            wb = list(raw.camera_whitebalance)
            if raw.color_desc == b'RGBG' and wb[3] == 0:
                wb[3] = wb[1]
        assert len(wb)==4 and np.isfinite(wb).all() and min(wb)>0
        captures = [[shot(rawpy,path,wb) for path in view.values()] for view in paths]
        colors = [entry[1]['matrix'] for view in captures for entry in view]
        sensor_keys = ('shape','black','white','pattern','flip','matrix_field','capture')
        sensor_agrees = all(all(view[0][1][key]==view[1][1][key] for key in sensor_keys) for view in captures)
        assert all(colors[0]==matrix for matrix in colors) and sensor_agrees
        components = [(view[0][0],view[1][0]) for view in captures]
        alignment = [registration(*pair) for pair in components]
        exposure = .70/max(float(np.quantile((a+b)@LUMA,.99)) for a,b in components)
        pairs = [(srgb8((a+b)*exposure),srgb8((a+b*GAIN)*exposure)) for a,b in components]
        hashes = {}
        for view, (base, cold) in enumerate(pairs,1):
            for role,image in [('base',base),('cold',cold)]:
                name = f'camera{view}-{role}.png'
                hashes[name] = save(folder/name,image)
        qualified = all(row['qualified'] for row in alignment)
        row = dict(sensors=[[entry[1] for entry in view] for view in captures], alignment=alignment,
                   source_clip=[[entry[2] for entry in view] for view in captures],
                   white_balance=wb, exposure=exposure, output_sha256=hashes, qualified=qualified)
        records[scene] = row
        if not qualified:
            continue
        (reference,cold_reference),(target,cold_target) = pairs
        anchor, cooled = [finish_spatial(target, engine.prepare(target, look).apply(target), .8)
                          for look in (reference,cold_reference)]
        for name,image in [('baseline-anchor.png',anchor),('baseline-cooled.png',cooled)]:
            hashes[name] = save(folder/name,image)
        truth = finish_spatial(target,cold_target,.8)
        delta = cooled.astype(np.int16)-anchor
        expected = truth.astype(np.int16)-target
        ambient,lamp = components[1]
        light = lamp@LUMA
        combined = (ambient+lamp)@LUMA
        fraction = np.divide(light,combined,out=np.zeros_like(light),where=combined>0)
        np.save(folder/'lamp-fraction.npy',fraction,allow_pickle=False)
        row['fraction_sha256'] = digest(folder/'lamp-fraction.npy')
        intensity = target@LUMA
        masks = dict(whole=np.ones(intensity.shape, bool), neutral=(np.ptp(target,axis=-1)<=20)&(intensity>=40)&(intensity<=220),
                     low=fraction<=.10, high=fraction>=.30)
        row['scores'] = scores(delta,expected,masks)
    assert time.monotonic()-start <= 120
    source = Path(__file__)
    result = dict(records=records, weight_sha256=engine.checkpoint_sha, rawpy=rawpy.__version__,
                  intake_sha256=digest(intake_path), source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('synclight-real-pixel-protocol.md')))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({scene: (row['alignment'],row.get('scores')) for scene,row in records.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decoder','sources','out'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    run(args.decoder,args.sources,args.out)
