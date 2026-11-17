"""Rebuild RGB features and verify grouped inference in the primal ridge form."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from triplet_color import to_srgb

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-reference-identity-v1'


def main():
    d=json.loads((OUT/'results.json').read_text());c=json.loads((OUT/'config.json').read_text())
    assert d['finished'] and d['rgb_only'] and not d['original_or_mask_images_read']
    assert digest(OUT/'config.json')==d['config_sha256']
    for n,h in c['source_sha256'].items():assert digest(OUT/'frozen-source'/n)==h
    old=json.loads((BASE/'ppr-pixel-sample-v1/results.json').read_text())
    more=json.loads((BASE/'ppr-expert-contrast-v1/results.json').read_text())
    files=[dict(r,expert='a') for r in old['files'] if r['role']=='target_a']+more['files']
    saved=np.load(OUT/'features.npz');features=[];moments=[]
    for row in d['rows']:
        item=next(r for r in files if r['id']==row['id'] and r['expert']==row['expert'])
        path=ROOT/item['local_path'];assert digest(path)==row['sha256']==item['sha256']
        with Image.open(path) as im:rgb,_=to_srgb(im);pixels=np.asarray(rgb.resize((256,256),Image.Resampling.BILINEAR))
        x=pixels.reshape(-1,3).astype(np.float64)/255
        mean=np.sum(x,axis=0)/len(x);std=np.sqrt(np.sum((x-mean)**2,axis=0)/len(x))
        summary=np.r_[mean,std]
        counts=np.concatenate([np.bincount(pixels[:,:,i].reshape(-1)//8,minlength=32)/65536 for i in range(3)])
        moments.append(summary);features.append(np.r_[counts,summary])
    features=np.asarray(features);moments=np.asarray(moments)
    assert np.array_equal(features,saved['primary']) and np.array_equal(moments,saved['summary'])
    labels=np.array([ord(r['expert'])-ord('a') for r in d['rows']]);groups=np.array([c['ids'].index(r['id']) for r in d['rows']])
    assert np.array_equal(labels,saved['labels']) and np.array_equal(groups,saved['groups'])
    perms=list(itertools.permutations(range(3)))
    permuted=np.asarray([[(0,1,2),*p] for p in itertools.product(perms,repeat=3)],dtype=int).reshape(216,12)
    null_hits=np.zeros(216,dtype=int);scores=np.empty((12,3));secondary=np.empty((12,3))
    for group in range(4):
        train=groups!=group;test=~train
        for featureset,output in [(features,scores),(moments,secondary)]:
            mean=featureset[train].mean(axis=0);std=featureset[train].std(axis=0);std[std==0]=1
            a=(featureset[train]-mean)/std;b=(featureset[test]-mean)/std
            operator=b@np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T)
            output[test]=operator@(np.eye(3)[labels[train]]-1/3)+1/3
            if featureset is features:
                predicted=np.einsum('ij,pjk->pik',operator,np.eye(3)[permuted[:,train]]-1/3)+1/3
                null_hits+=(predicted.argmax(axis=2)==permuted[:,test]).sum(axis=1)
    predicted=scores.argmax(axis=1);hits=int((predicted==labels).sum());p=float((null_hits>=hits).mean())
    assert hits==d['main_correct'] and int((secondary.argmax(axis=1)==labels).sum())==d['secondary_correct']
    assert np.array_equal(null_hits,d['exact_permutation_counts']) and p==d['exact_p']
    expected=np.array([r['ridge_scores'] for r in d['rows']]);error=float(np.abs(scores-expected).max());assert error<1e-9
    assert [r['predicted_expert'] for r in d['rows']]==['abc'[i] for i in predicted]
    assert d['reference_identity_signal']==bool(hits>=9 and p<=.05)
    report=dict(finished=True,results_sha256=digest(OUT/'results.json'),driver_sha256=digest(__file__),expert_images_redecoded=12,feature_values_exact=True,primal_dual_max_score_difference=error,permutations_recomputed=216,exact_p=p,main_correct=hits,signal_unchanged=True,original_or_mask_images_read=False,software_acceptance=False)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
