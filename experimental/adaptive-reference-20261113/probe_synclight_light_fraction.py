"""Test whether light-fraction labels transfer across the two captured rooms."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from audit_bg_real_skin_regions import BASE, ROOT, digest
from audit_synclight_existing_direction import BASELINE_SHA
from probe_synclight_warm_support import warmth_weight


LUMA = np.array([.2126,.7152,.0722])


def features(rgb):
    values = rgb.astype(float)/255
    linear = np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2.4)
    red,green,blue = np.moveaxis(np.log(linear+.01),-1,0)
    channels = [np.log(linear@LUMA+.01),red-green,blue-green]
    size = (rgb.shape[1],rgb.shape[0])
    smooth = [np.asarray(Image.fromarray(value.astype(np.float32)).resize((32,24),Image.Resampling.BOX)
                         .resize(size,Image.Resampling.BILINEAR)) for value in channels]
    return np.stack(channels+smooth,axis=-1)


def fit(train, truth, target):
    x = train[::4,::4].reshape(-1,train.shape[-1])
    y = truth[::4,::4].ravel()
    mean,scale = x.mean(axis=0),x.std(axis=0)
    assert np.all(scale>0) and np.isfinite(y).all()
    normalized = (x-mean)/scale
    beta = np.linalg.solve(normalized.T@normalized+.01*len(y)*np.eye(x.shape[1]),
                           normalized.T@(y-y.mean()))
    prediction = ((target-mean)/scale)@beta+y.mean()
    return prediction.clip(0,1),dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),
                                    intercept=float(y.mean()),samples=len(y))


def performance(strong, weak, selections):
    assert strong.any() and weak.any()
    high = float(np.mean(selections[strong]))
    low = float(np.mean(~selections[weak]))
    return dict(strong_correct=high,weak_correct=low,balanced_accuracy=(high+low)/2)


def run(out):
    start=time.monotonic()
    evidence=ROOT/'docs/research/2026-11-18-synclight-ambient-balance-evidence.json'
    assert digest(evidence)==BASELINE_SHA
    data=json.loads(evidence.read_bytes())
    parent=BASE/'synclight-ambient-balance'
    out.mkdir()
    images, covariates, labels = {}, {}, {}
    for scene,row in data['records'].items():
        rgb=parent/scene/'camera2-base.png'
        labels[scene]=parent/scene/'lamp-fraction.npy'
        assert digest(rgb)==row['output_sha256'][rgb.name]
        assert digest(labels[scene])==row['fraction_sha256'] and row['local_eligible']
        with Image.open(rgb) as image:
            images[scene]=np.array(image.convert('RGB'))
        covariates[scene]=features(images[scene])
    assert len(images)==2
    records={}
    for target in images:
        train=next(key for key in images if key!=target)
        train_truth=np.load(labels[train],allow_pickle=False)
        pred, model=fit(covariates[train],train_truth,covariates[target])
        path=out/f'{target}-fraction.npy'
        np.save(path,pred,allow_pickle=False)
        actual=np.load(labels[target],allow_pickle=False)
        strong,weak=actual>=.30,actual<=.10
        assert min(strong.mean(),weak.mean())>=.05
        measured=performance(strong,weak,pred>=.20)
        warmth=performance(strong,weak,warmth_weight(images[target])>=1)
        error=float(np.abs(pred-actual).mean())
        control=float(np.abs(model['intercept']-actual).mean())
        passed=(measured['balanced_accuracy']>=.75 and
                measured['balanced_accuracy']>=warmth['balanced_accuracy']+.05 and error<=.85*control)
        records[target]=dict(training_room=train,regression=model,supervised=measured,warmth=warmth,mae=error,
                             constant_mae=control,passed=passed,prediction_sha256=digest(path))
    assert time.monotonic()-start<=60
    source=Path(__file__)
    result=dict(records=records,proceed=all(row['passed'] for row in records.values()),
                parent_sha256=BASELINE_SHA,source_sha256=digest(source),
                protocol_sha256=digest(source.with_name('synclight-light-fraction-study-protocol.md')))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print({key:(row['supervised'],row['warmth'],row['mae'],row['constant_mae'],row['passed'])
           for key,row in records.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
