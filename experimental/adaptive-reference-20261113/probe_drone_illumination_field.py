"""Check spatial interpolation of gray-ball illumination by held-out image tile."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe_drone_wall_illuminant import PAINT, SCENES, label_rows, sha


def angles(actual, expected):
    cross = np.linalg.norm(np.cross(actual,expected),axis=-1)
    dot = np.sum(actual*expected,axis=-1)
    return np.rad2deg(np.arctan2(cross,dot))


def estimate(points, colors, places, short_edge):
    squared = np.square((places[:,None,:]-points[None,:,:])/short_edge).sum(-1)
    nearest = np.argsort(squared,axis=1,kind='stable')[:,:4]
    distance = np.take_along_axis(squared,nearest,axis=1)
    support = distance[:,-1] <= .25**2
    weights = 1/np.maximum(distance,1e-12)
    predicted = np.sum(weights[...,None]*colors[nearest],axis=1)/weights.sum(1,keepdims=True)
    return predicted,support,np.sqrt(distance[:,-1])


def comparison(points, colors, shape):
    h,w = shape
    tiles = points*4//np.array([w,h])
    errors = np.zeros(len(points))
    controls = np.zeros(len(points))
    support = np.zeros(len(points),dtype=bool)
    distances = np.zeros(len(points))
    for tile in np.unique(tiles,axis=0):
        held = (tiles==tile).all(-1)
        control = colors[~held].mean(0)
        controls[held] = angles(control,colors[held])
        local,covered,far = estimate(points[~held],colors[~held],points[held],min(h,w))
        errors[held] = np.where(covered,angles(local,colors[held]),controls[held])
        support[held],distances[held] = covered,far
    return errors,controls,support,distances


def percentile(errors):
    return np.quantile(errors,[.5,.9]).tolist() if len(errors) else None


def decision(errors,control,supported):
    if not supported.any():
        return dict(coverage=0.,supported=None,all_points=dict(local=percentile(errors),global_=percentile(control)),
                    passed=False)
    local,global_ = (percentile(values[supported]) for values in (errors,control))
    coverage = float(supported.mean())
    passed = coverage>=.8 and global_[0]-local[0]>=.5 and local[0]<=.8*global_[0] and local[1]<=global_[1]+1
    return dict(coverage=coverage,supported=dict(local=local,global_=global_),
                all_points=dict(local=percentile(errors),global_=percentile(control)),passed=bool(passed))


def support_preview(source, out, points, errors, controls, support, shape):
    h,w = shape
    with Image.open(source) as image:
        im = image.convert('RGB')
    mark = ImageDraw.Draw(im)
    for (x,y),e,c,valid in zip(points,errors,controls,support,strict=True):
        u,v = x*im.width/w,y*im.height/h
        color = '#32e366' if valid and e<c else '#f24e58' if valid else '#ededed'
        mark.ellipse((u-3,v-3,u+3,v+3),fill=color,outline='#2a2a2a')
    font = ImageFont.load_default(size=13)
    wrapped = Image.new('RGB',(im.width,im.height+25),'#191c20')
    wrapped.paste(im,(0,25))
    ImageDraw.Draw(wrapped).text((5,5),'Green: lower error   Red: equal or higher   White: insufficient support',
                                 font=font,fill='#f6f7f8')
    wrapped.save(out,quality=92,subsampling=0)


def show_correction(raw_file, out, points, colors):
    import rawpy

    with rawpy.imread(str(raw_file)) as raw:
        sensor = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                 output_color=rawpy.ColorSpace.raw,output_bps=16,gamma=(1,1),
                                 user_wb=[1,1,1,1],no_auto_bright=True,user_flip=0)
    h,w = sensor.shape[:2]
    rgb = sensor[::8,::8].astype(float)
    places = np.column_stack([a.ravel()*8 for a in np.indices(rgb.shape[:2])[::-1]])
    control = colors.mean(0)
    values = []
    for start in range(0,len(places),4096):
        candidate,yes,_ = estimate(points,colors,places[start:start+4096],min(h,w))
        values.append(np.where(yes[:,None],candidate,control))
    local = np.concatenate(values).reshape(rgb.shape)

    def white_balance(illuminant):
        adjusted = rgb/(illuminant*PAINT)
        return adjusted*np.divide(rgb.sum(-1),adjusted.sum(-1),out=np.ones(rgb.shape[:2]),
                                  where=adjusted.sum(-1)>0)[...,None]

    scale = max(float(np.quantile(rgb,.99)),1.)
    corrected = np.concatenate([white_balance(x) for x in (control,local)],axis=1)
    photo = np.rint((corrected/scale).clip(0,1)**(1/2.2)*255).astype(np.uint8)
    Image.fromarray(photo).save(out,quality=91,subsampling=0)


def run(source, previous, decoder, evidence, out):
    start = time.monotonic()
    validated = json.loads(evidence.read_bytes())
    known = {row['name']:row['sha256'] for row in validated['source_files']}
    out.mkdir()
    sys.path.insert(0,str(decoder.resolve()))
    result = {}
    for name in SCENES:
        old = validated['scenes'][name]
        csv = source/f'{name}.csv'
        original_preview = previous/f'{name}.jpg'
        if not old['qualified'] or sha(csv)!=known[csv.name] or sha(original_preview)!=old['preview_sha256']:
            raise ValueError('Saved development source changed')
        shape = old['dimensions'][::-1]
        labels,_,_,_ = label_rows(csv,shape)
        if len(labels)!=old['samples']:
            raise ValueError('Label count changed')
        points = np.array([(x,y) for _,x,y,_,_ in labels])
        colors = np.array([c for _,_,_,_,c in labels])
        error,control,support,distances = comparison(points,colors,shape)
        stats = decision(error,control,support)
        preview = out/f'{name}-support.jpg'
        support_preview(original_preview,preview,points,error,control,support,shape)
        np.savez_compressed(out/f'{name}-points.npz',local=error,global_control=control,support=support,distances=distances)
        panel = None
        if stats['passed']:
            raw_file = source/f'{name}.NEF'
            if sha(raw_file)!=known[raw_file.name]:
                raise ValueError('Saved RAW source changed')
            image = out/f'{name}-controlled-grade.jpg'
            show_correction(raw_file,image,points,colors)
            panel = sha(image)
        result[name] = dict(stats,samples=len(points),supported_points=int(support.sum()),
                            support_sha256=sha(preview),controlled_grade_sha256=panel)
        print(name,result[name])
    if time.monotonic()-start>120:
        raise RuntimeError('Run exceeded compute budget')
    script = Path(__file__)
    report = dict(scenes=result,parent_sha256=sha(evidence),source_sha256=sha(script),
                  protocol_sha256=sha(script.with_name('drone-illumination-field-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source','previous','decoder','evidence','out'):
        parser.add_argument(key,type=Path)
    run(**vars(parser.parse_args()))
