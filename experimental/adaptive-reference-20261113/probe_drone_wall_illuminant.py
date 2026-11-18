"""Verify spatial gray-ball labels and audit a fixed local illumination estimate."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENES = ('dining_area','wall_hallway')
PAINT = np.array([.22464097,.4464182,.32894083])
COLOR = ('red_channel','green_channel','blue_channel')
GEOMETRY = ('circle_x','circle_y','circle_radiuos')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def geometry(prefix):
    if prefix[:4] != b'II*\x00':
        raise ValueError('Expected an uncompressed little-endian author TIFF')
    pos, = struct.unpack_from('<I',prefix,4)
    count, = struct.unpack_from('<H',prefix,pos)
    entries = {}
    for index in range(count):
        tag,kind,n,value = struct.unpack_from('<HHII',prefix,pos+2+12*index)
        if n == 1 and kind in (3,4):
            entries[tag] = value & 65535 if kind == 3 else value
        if tag == 258 and kind == 3 and n == 3:
            if struct.unpack_from('<3H',prefix,value) != (16,16,16):
                raise ValueError('Expected three 16-bit channels')
    width,height = entries[256],entries[257]
    start = entries[273]
    if (entries[259],entries[262],entries[277],entries[278],entries[284]) != (1,2,3,height,1):
        raise ValueError('Unexpected TIFF strip layout')
    if entries.get(274,1) != 1 or entries[279] != width*height*6:
        raise ValueError('Unexpected TIFF orientation or size')
    return width,height,start


def audit_axis(rgb, prefix, center, width, start):
    first_rows = np.frombuffer(prefix,dtype='<u2',count=width*6,offset=start).reshape(2,width,3)
    mid_rows = np.frombuffer(center,dtype='<u2').reshape(2,width,3)
    result = []
    for native,author in ((rgb[:2],first_rows),(rgb[1421:1423],mid_rows)):
        gradients = [np.diff(x.mean(2),axis=1)[:,2:-2].ravel() for x in (native,author)]
        corr = float(np.corrcoef(gradients)[0,1])
        result.append(corr)
    return result


def label_rows(path, shape):
    height,width = shape
    valid,rejected = [],[]
    with path.open(newline='',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            color = np.array([float(row[k]) for k in COLOR])
            x,y,radius = [int(row[k]) for k in GEOMETRY]
            name = row['image_name']
            if np.isfinite(color).all() and (color>0).all() and radius>0 and 0<=x<width and 0<=y<height:
                valid.append((name,x,y,radius,color/color.sum()))
            else:
                rejected.append(name)
    if len({v[0] for v in valid}) != len(valid):
        raise ValueError('Duplicate source frame key')
    coords = {(x,y) for _,x,y,_,_ in valid}
    tiles = {(4*x//width,4*y//height) for x,y in coords}
    return valid,rejected,len(coords),len(tiles)


def angle(a,b):
    cos = (a@b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.rad2deg(np.arccos(np.clip(cos,-1,1))))


def profile(rgb, labels):
    h,w = rgb.shape[:2]
    radius = min(h,w)//8
    constant = rgb.mean(axis=(0,1))/PAINT
    worst = []
    for name,x,y,r,truth in labels:
        region = rgb[max(y-radius,0):min(y+radius+1,h),max(x-radius,0):min(x+radius+1,w)]
        local = region.mean(axis=(0,1))/PAINT
        worst.append(dict(frame=name,xy=[x,y],radius=r,truth=truth.tolist(),
                          local=(local/local.sum()).tolist(),
                          local_degrees=angle(local,truth),full_degrees=angle(constant,truth)))
    errors = {label:[float(np.quantile([r[key] for r in worst],q)) for q in (.5,.9)]
              for label,key in (('local','local_degrees'),('global','full_degrees'))}
    passed = errors['local'][0] <= .8*errors['global'][0] and errors['local'][1] <= errors['global'][1]+1
    return dict(errors=errors,records=worst,window_radius=radius,full_normalized=(constant/constant.sum()).tolist(),
                passed=bool(passed))


def preview(rgb, labels, path):
    h,w = rgb.shape[:2]
    thumb = rgb[::8,::8].astype(np.float32)
    scale = max(float(np.quantile(thumb,.99)),1.)
    im = Image.fromarray(np.rint((thumb/scale).clip(0,1)**(1/2.2)*255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    for _,x,y,r,_ in labels:
        u,v = x*im.width/w,y*im.height/h
        draw.ellipse((u-2,v-2,u+2,v+2), fill='#ffb31b')
    draw.text((8,7),f'{len(labels)} illumination measurements',font=ImageFont.load_default(size=16),
              fill='#fff2d9',stroke_width=1,stroke_fill='#171717')
    im.save(path,quality=91,subsampling=0)


def run(data, decoder, out):
    start = time.monotonic()
    identity = json.loads((data/'intake.json').read_text(encoding='utf-8-sig'))
    for record in identity:
        path = data/record['name']
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Source changed: {path}')
    if sum(row['bytes'] for row in identity if row['name'].endswith('.NEF')) > 32*1024**2:
        raise ValueError('Media budget exceeded')
    if (data/'conversion.txt').read_text().strip() != 'dcrawMS -v -q 0 -o 0 -4 -r 1 1 1 1 -T test.NEF':
        raise ValueError('Author conversion changed')
    sys.path.insert(0,str(decoder.resolve()))
    import rawpy
    if rawpy.__version__ != '0.28.0':
        raise ValueError('Expected the pinned local RAW decoder')
    out.mkdir()
    results = {}
    for scene in SCENES:
        path = data/f'{scene}.NEF'
        with Image.open(path) as info:
            if info.getexif().get(272) != 'NIKON D700' or info.getexif().get(274) != 1:
                raise ValueError('Unexpected source camera or orientation')
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                  output_color=rawpy.ColorSpace.raw,output_bps=16,
                                  gamma=(1,1),user_wb=[1,1,1,1],no_auto_bright=True,user_flip=0)
        tiff = data/f'{scene}.tiff.head'
        rows = data/f'{scene}.center-row'
        head = tiff.read_bytes()
        width,height,offset = geometry(head)
        if rgb.shape != (height,width,3) or rgb.dtype != np.uint16:
            raise ValueError(f'Native crop differs from author dimensions for {scene}')
        axis = audit_axis(rgb,head,rows.read_bytes(),width,offset)
        labels,rejected,unique,tiles = label_rows(data/f'{scene}.csv',(height,width))
        ready = min(axis)>=.95 and len(labels)>=20 and unique>=20 and tiles>=4
        image = out/f'{scene}.jpg'
        preview(rgb,labels,image)
        results[scene] = dict(dimensions=[width,height],scanline_correlations=axis,samples=len(labels),
                              rejected=rejected,unique_centers=unique,occupied_tiles=tiles,qualified=ready,
                              tiff_header_sha256=sha(tiff),tiff_center_sha256=sha(rows),preview_sha256=sha(image),
                              estimates=profile(rgb,labels) if ready else None)
        print(scene,axis,len(labels),tiles,results[scene]['estimates'] and results[scene]['estimates']['errors'])
    if time.monotonic()-start > 120:
        raise RuntimeError('Run exceeded compute budget')
    source = Path(__file__)
    report = dict(scenes=results,source_files=identity,decoder=rawpy.__version__,paint=PAINT.tolist(),
                  source_sha256=sha(source),protocol_sha256=sha(source.with_name('drone-wall-illuminant-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('data','decoder','out'):
        parser.add_argument(arg,type=Path)
    run(**vars(parser.parse_args()))
