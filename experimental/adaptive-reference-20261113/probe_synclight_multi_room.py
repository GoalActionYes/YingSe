"""Repeat the fixed light-fraction screen with more distinct training rooms."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from audit_bg_real_skin_regions import BASE, ROOT, digest
from audit_synclight_existing_direction import BASELINE_SHA
from probe_synclight_ambient_balance import fraction, reconstruct_controls
from probe_synclight_light_fraction import features, performance
from probe_synclight_real_pixels import registration, shot
from probe_synclight_warm_support import warmth_weight

KEYS = ('shape','black','white','pattern','flip','matrix_field','matrix','capture')


def prepare(decoder, sources, out):
    started=time.monotonic()
    sys.path.insert(0,str(decoder.resolve()))
    import rawpy

    assert rawpy.__version__=='0.28.0'
    intake_path=sources/'intake.json'
    intake=json.loads(intake_path.read_bytes())
    assert intake['selected']==['scene22','scene18','scene37','scene24']
    out.mkdir()
    result={}
    for scene in intake['selected']:
        if scene in intake['omitted']:
            result[scene]=dict(qualified=False, reason='missing fixed DNG pair')
            continue
        paths=[sources/f'{scene}_1_{role}.DNG' for role in ('0_ambient','1_camvis')]
        assert all(digest(p)==intake['files'][p.name]['sha256'] for p in paths)
        with rawpy.imread(str(paths[0])) as raw:
            wb=list(raw.camera_whitebalance)
            if raw.color_desc==b'RGBG' and wb[3]==0:
                wb[3]=wb[1]
        assert len(wb)==4 and min(wb)>0
        first,second=[shot(rawpy,path,wb) for path in paths]
        capture_match=all(first[1][key]==second[1][key] for key in KEYS)
        ambient,lamp=first[0],second[0]
        aligned=registration(ambient,lamp) if capture_match else None
        row=dict(capture_match=capture_match,alignment=aligned,qualified=False,white_balance=wb)
        result[scene]=row
        if aligned is None or not aligned['qualified']:
            continue
        ratio,_=fraction(ambient,lamp,1)
        q=float(np.quantile(ratio,.05,method='higher'))
        if not 0 < q < 1:
            row['reason']='degenerate fifth percentile'
            continue
        mult=math.ceil(9*q/(1-q))
        light,_=fraction(ambient,lamp,mult)
        coverage=[float((light<=.10).mean()),float((light>=.30).mean())]
        pair,exposure=reconstruct_controls([(ambient,lamp)],mult)
        png=out/f'{scene}.png'
        labels=out/f'{scene}.npy'
        Image.fromarray(pair[0][0]).save(png)
        np.save(labels,light,allow_pickle=False)
        row.update(multiplier=mult,coverage=coverage,exposure=exposure,qualified=min(coverage)>=.05,
                   png_sha256=digest(png),label_sha256=digest(labels))
    assert time.monotonic()-started<=120
    report=dict(records=result,intake_sha256=digest(intake_path))
    (out/'prepared.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({name:(row['qualified'],row.get('coverage'),row.get('alignment')) for name,row in result.items()})


def fit(train_dir,out):
    started=time.monotonic()
    prepared=train_dir/'prepared.json'
    decoded=json.loads(prepared.read_bytes())
    rows=decoded['records']
    eligible=[key for key,row in rows.items() if row['qualified']]
    if len(eligible)<3:
        raise ValueError('Fewer than three eligible training rooms')
    doc=ROOT/'docs/research/2026-11-18-synclight-ambient-balance-evidence.json'
    assert digest(doc)==BASELINE_SHA
    held=json.loads(doc.read_bytes())['records']
    assert not set(rows)&set(held)
    x,y=[],[]
    for key in eligible:
        row=rows[key]
        rgb=train_dir/f'{key}.png'
        mask=train_dir/f'{key}.npy'
        assert digest(rgb)==row['png_sha256'] and digest(mask)==row['label_sha256']
        with Image.open(rgb) as image:
            x.append(features(np.array(image))[::4,::4].reshape(-1,6))
        y.append(np.load(mask,allow_pickle=False)[::4,::4].ravel())
    x,y=np.concatenate(x),np.concatenate(y)
    assert len(x)==len(y)
    mean,std=x.mean(axis=0),x.std(axis=0)
    normalized=(x-mean)/std
    model=np.linalg.solve(normalized.T@normalized+.01*len(y)*np.eye(x.shape[1]),
                          normalized.T@(y-y.mean()))
    out.mkdir()
    results={}
    for name,row in held.items():
        folder=BASE/'synclight-ambient-balance'/name
        rgb=folder/'camera2-base.png'
        assert digest(rgb)==row['output_sha256'][rgb.name]
        with Image.open(rgb) as image:
            original=np.array(image)
        prediction=(((features(original)-mean)/std)@model+y.mean()).clip(0,1)
        result=out/f'{name}.npy'
        np.save(result,prediction,allow_pickle=False)
        labels=folder/'lamp-fraction.npy'
        assert digest(labels)==row['fraction_sha256']
        expected=np.load(labels,allow_pickle=False)
        strong,weak=expected>=.30,expected<=.10
        metrics=performance(strong,weak,prediction>=.20)
        warm=performance(strong,weak,warmth_weight(original)>=1)
        mae=float(np.abs(prediction-expected).mean())
        control=float(np.abs(y.mean()-expected).mean())
        ok=metrics['balanced_accuracy']>=.75 and metrics['balanced_accuracy']>=warm['balanced_accuracy']+.05
        results[name]=dict(learned=metrics,warmth=warm,mae=mae,constant_mae=control,
                           passed=ok and mae<=.85*control,prediction_sha256=digest(result))
    assert time.monotonic()-started<=120
    source=Path(__file__)
    record=dict(eligible_training_rooms=eligible,sampled_pixels=len(y),training=rows,held=results,
                passed=all(row['passed'] for row in results.values()),prepared_sha256=digest(prepared),
                parent_sha256=BASELINE_SHA,source_sha256=digest(source),
                protocol_sha256=digest(source.with_name('synclight-multi-room-protocol.md')))
    (out/'results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='phase',required=True)
    preparation=sub.add_parser('prepare')
    for name in ('decoder','sources','out'):
        preparation.add_argument(name,type=Path)
    training=sub.add_parser('fit')
    for name in ('train','out'):
        training.add_argument(name,type=Path)
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare(args.decoder,args.sources,args.out)
    else:
        fit(args.train,args.out)
