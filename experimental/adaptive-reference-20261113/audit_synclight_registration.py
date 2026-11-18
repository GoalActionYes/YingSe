"""Audit capture registration with exposure-normalized structural correspondences."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from audit_bg_real_skin_regions import BASE, digest

SCENES = ('scene22','scene37','scene18','scene36')


def decode(decoder, out):
    from probe_synclight_multi_room import KEYS
    from probe_synclight_real_pixels import LUMA, shot, srgb8

    started=time.monotonic()
    sys.path.insert(0,str(decoder.resolve()))
    import rawpy

    assert rawpy.__version__=='0.28.0'
    older=BASE/'synclight-multi-room-input'
    newer=BASE/'synclight-extra-rooms-input'
    groups={folder:json.loads((folder/'intake.json').read_bytes())['files'] for folder in (older,newer)}
    out.mkdir()
    scenes={}
    for scene in SCENES:
        folder=older if scene in ('scene22','scene37') else newer
        files=groups[folder]
        chosen={Path(name).stem:name for name in files if name.startswith(scene+'_1_')}
        paths=[folder/chosen[f'{scene}_1_{role}'] for role in ('0_ambient','1_camvis')]
        ids={path.name:digest(path) for path in paths}
        assert all(sha==files[name]['sha256'] for name,sha in ids.items())
        with rawpy.imread(str(paths[0])) as raw:
            wb=list(raw.camera_whitebalance)
            if raw.color_desc==b'RGBG' and wb[3]==0:
                wb[3]=wb[1]
        first,second=[shot(rawpy,path,wb) for path in paths]
        assert all(first[1][key]==second[1][key] for key in KEYS)
        hashes={}
        for name,image in [('ambient',first[0]),('lamp',second[0])]:
            quantile=float(np.quantile(image@LUMA,.99))
            assert quantile>0
            path=out/f'{scene}-{name}.png'
            Image.fromarray(srgb8(image*(.70/quantile))).save(path)
            hashes[path.name]=digest(path)
        scenes[scene]=dict(raw=ids,output=hashes)
    assert time.monotonic()-started<=90
    (out/'decode.json').write_text(json.dumps(scenes,indent=2)+'\n',encoding='utf-8')


def match(images,out):
    import cv2
    from probe_synclight_cross_view import accepted_points, load

    started=time.monotonic()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    sift=cv2.SIFT_create()
    identity=images/'decode.json'
    manifest=json.loads(identity.read_bytes())
    pictures={}
    for scene in SCENES:
        pictures[scene]=[load(images/name,expected) for name,expected in manifest[scene]['output'].items()]

    def compare(a,b):
        data,points=accepted_points(sift,a,b)
        if points is None:
            return dict(**data,shift_median=None,shift_p90=None,same_coordinates=False)
        shift=np.linalg.norm(points[1]-points[0],axis=1)
        median,p90=map(float,np.quantile(shift,[.5,.9]))
        return dict(**data,shift_median=median,shift_p90=p90,same_coordinates=median<=.75 and p90<=1.5)

    same={scene:compare(*pictures[scene]) for scene in SCENES}
    wrong={}
    for scene in ('scene22','scene37'):
        first,second=pictures[scene]
        for axis in (0,1):
            wrong[f'{scene}-shift-{axis}']=compare(first,np.roll(second,4,axis=axis))
        other='scene37' if scene=='scene22' else 'scene22'
        wrong[f'{scene}-room']=compare(first,pictures[other][1])
    valid=(all(same[name]['same_coordinates'] for name in ('scene22','scene37'))
           and not any(row['same_coordinates'] for row in wrong.values()))
    potential=[key for key in ('scene18','scene36') if same[key]['same_coordinates']] if valid else []
    assert time.monotonic()-started<=90
    source=Path(__file__)
    result=dict(same=same,negative=wrong,control_pass=valid,potential=potential,
                decode_sha256=digest(identity),opencv=cv2.__version__,source_sha256=digest(source),
                protocol_sha256=digest(source.with_name('synclight-registration-audit-protocol.md')))
    with out.open('x',encoding='utf-8') as handle:
        handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(valid,potential,same)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('decode','match'))
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    if args.phase=='decode':
        decode(args.source,args.out)
    else:
        match(args.source,args.out)
