"""Screen geometric correspondence for a new local color response."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from audit_bg_real_skin_regions import BASE, ROOT, digest


LUMA = np.array([.2126,.7152,.0722])
PIN = '558e2b12f012c59314f30f4b0df165b565bca6ffe1ac5482ffbacac52a570afb'


def load(path, identity):
    assert digest(path)==identity
    with Image.open(path) as picture:
        return np.array(ImageOps.exif_transpose(picture).convert('RGB'))


def detect(sift, rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    return sift.detectAndCompute(gray,None)


def accepted_points(sift, target, reference):
    p,a=detect(sift,target)
    q,b=detect(sift,reference)
    status=dict(keypoints=[len(p),len(q)], mutual=0, inliers=0, fraction=0., coverage=[0.,0.], accepted=False)
    if min(len(p),len(q))<2:
        return status,None
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def filtered(first, second):
        return {best.queryIdx:best.trainIdx for best, runner_up in matcher.knnMatch(first,second,k=2)
                if best.distance<.75*runner_up.distance}
    forward,backward=filtered(a,b),filtered(b,a)
    pairs=[(x,y) for x,y in forward.items() if backward.get(y)==x]
    status['mutual']=len(pairs)
    if len(pairs)<4:
        return status,None
    first=np.float32([p[x].pt for x,_ in pairs])
    second=np.float32([q[y].pt for _,y in pairs])
    _,inliers=cv2.findHomography(first,second,cv2.RANSAC,3.)
    if inliers is None:
        return status,None
    keep=inliers.ravel().astype(bool)
    count=int(keep.sum())
    status.update(inliers=count,fraction=count/len(pairs))
    points=first[keep],second[keep]
    coverage=[float(cv2.contourArea(cv2.convexHull(pts))/(rgb.shape[0]*rgb.shape[1]))
              if count>=3 else 0. for pts,rgb in zip(points,(target,reference),strict=True)]
    passes=count>=12 and status['fraction']>=.5 and min(coverage)>=.08
    status.update(coverage=coverage,accepted=passes)
    return status,points if passes else None


def chroma(rgb):
    return rgb-(rgb@LUMA)[...,None]


def patch_median(rgb, point):
    x,y=map(int,np.rint(point))
    return np.median(rgb[max(0,y-4):y+5,max(0,x-4):x+5].reshape(-1,3),axis=0)


def transfer(sift,target,reference,baseline):
    status,points=accepted_points(sift,target,reference)
    if points is None:
        return baseline.copy(),status
    total=np.zeros(target.shape[:2])
    weighted=np.zeros(target.shape,dtype=float)
    yy,xx=np.ogrid[:target.shape[0],:target.shape[1]]
    for position,correspondence in zip(*points,strict=True):
        x,y=position
        distance=np.sqrt((xx-x)**2+(yy-y)**2)
        weight=np.maximum(1-distance/36,0)
        vector=chroma(patch_median(reference,correspondence)-patch_median(target,position))
        weighted+=weight[...,None]*vector
        total+=weight
    correction=np.divide(weighted,total[...,None],where=total[...,None]>0,out=np.zeros_like(weighted))
    alpha=np.minimum(total,1)[...,None]
    new_chroma=chroma(target)+.8*correction
    combined=(baseline@LUMA)[...,None]+alpha*new_chroma+(1-alpha)*chroma(baseline)
    output=np.rint(combined.clip(0,255)).astype(np.uint8)
    return output,status


def metrics(current, answer, mask):
    x,y=current[mask].astype(float),answer[mask].astype(float)
    power=np.square(y).sum()
    return dict(pixels=int(mask.sum()),mae=float(np.abs(x-y).mean()),
                signal=float(np.abs(y).mean()),direction=float(np.sum(x*y)/power) if power else None)


def run(out):
    start=time.monotonic()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    sift=cv2.SIFT_create()
    doc=ROOT/'docs/research'
    pinned=doc/'2026-11-18-synclight-ambient-balance-evidence.json'
    assert digest(pinned)==PIN
    rooms=json.loads(pinned.read_bytes())['records']
    old=doc/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    held=json.loads(old.read_bytes())
    parent=BASE/'background-intervention-20261116-v2'
    manifest=parent/'results.json'
    assert digest(manifest)==held['parent_sha256']
    index={case['pair']:case['input_file'] for case in json.loads(manifest.read_bytes())['cases']}
    source=BASE/'synclight-ambient-balance'
    out.mkdir()
    output={}
    saved={}
    geometry={}
    positive={}
    negatives={}
    for scene,row in rooms.items():
        read=lambda name:load(source/scene/name,row['output_sha256'][name])
        target=read('camera2-base.png')
        saved[scene]={}
        for name,reference in [('anchor','camera1-base.png'),('cooled','camera1-cold.png')]:
            base=read(f'baseline-{name}.png')
            result,status=transfer(sift,target,read(reference),base)
            image=out/f'{scene}-{name}.png'
            Image.fromarray(result).save(image)
            output[image.name]=digest(image)
            saved[scene][name]=result
            geometry[f'{scene}-{name}']=status
        positive[scene]=accepted_points(sift,read('camera1-base.png'),read('camera1-cold.png'))[0]
        other=next(key for key in rooms if key!=scene)
        path=source/other/'camera1-cold.png'
        neg=load(path,rooms[other]['output_sha256'][path.name])
        negatives[f'{scene}-other']=accepted_points(sift,target,neg)[0]
    for key,row in held['cases'].items():
        source_file=index[row['target_pair']]
        target=load(parent/source_file['path'],source_file['sha256'])
        for role in ('C1','F1'):
            info=row['input'][role]
            reference=load(BASE/'bg-real-pixel-inputs'/info['name'],info['sha256'])
            negatives[f'{key}-{role}']=accepted_points(sift,target,reference)[0]
    room_scores={}
    for scene,row in rooms.items():
        read=lambda name:load(source/scene/name,row['output_sha256'][name]).astype(np.int16)
        target=read('camera2-base.png')
        truth=(4*read('camera2-cold.png')+target+2)//5-target
        delta=saved[scene]['cooled'].astype(np.int16)-saved[scene]['anchor']
        path=source/scene/'lamp-fraction.npy'
        assert digest(path)==row['fraction_sha256']
        fraction=np.load(path,allow_pickle=False)
        y=target@LUMA
        scopes=dict(high=fraction>=.30,low=fraction<=.10,
                    neutral=(np.ptp(target,axis=-1)<=20)&(y>=40)&(y<=220))
        score={key:metrics(delta,truth,mask) for key,mask in scopes.items() if mask.sum()>=256}
        high,low=(score[key] for key in ('high','low'))
        passed=low['mae']<=1 and .75<=high['direction']<=1.25 and high['mae']<=.85*high['signal']
        room_scores[scene]=dict(scopes=score,passed=passed)
    gates=dict(rooms=all(case['passed'] for case in room_scores.values()),
               neutral=room_scores['scene25']['scopes']['neutral']['mae']<=rooms['scene25']['scores']['neutral']['mae']+.25,
               negative=not any(case['accepted'] for case in negatives.values()),
               same_view=all(case['accepted'] for case in positive.values()))
    assert len(negatives)==10 and time.monotonic()-start<=60
    details=dict(geometry=geometry,positive=positive,negative=negatives,rooms=room_scores,gates=gates,
                 proceed=all(gates.values()),output_sha256=output,opencv=cv2.__version__,source_sha256=digest(Path(__file__)),
                 parent_sha256=PIN,bg_sha256=digest(old))
    (out/'results.json').write_text(json.dumps(details,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(geometry,gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
