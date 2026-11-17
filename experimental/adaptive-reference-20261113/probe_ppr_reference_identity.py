"""Reference-only grouped label probe, with an exact groupwise permutation test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import itertools
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from triplet_color import to_srgb

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-reference-identity-v1'


def transform(rgb):
    x=rgb.astype(np.float64)/255
    summary=np.r_[x.mean(axis=(0,1)),x.std(axis=(0,1))]
    hist=np.concatenate([np.histogram(rgb[:,:,c],bins=32,range=(0,256))[0]/(256*256) for c in range(3)])
    return summary,np.r_[hist,summary]


def operators(features,groups):
    folds=[]
    for group in range(4):
        train=groups!=group;test=groups==group
        mean=features[train].mean(axis=0);std=features[train].std(axis=0);std[std==0]=1
        x=(features[train]-mean)/std;t=(features[test]-mean)/std
        # Equivalent dual form of fixed ridge regression; standardization uses training only.
        op=t@x.T@np.linalg.solve(x@x.T+np.eye(9),np.eye(9))
        folds.append((train,test,op))
    return folds


def predict(folds,labels):
    scores=np.empty((12,3),dtype=np.float64)
    for train,test,op in folds:scores[test]=op@(np.eye(3)[labels[train]]-1/3)+1/3
    return scores.argmax(axis=1),scores


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    old_path=BASE/'ppr-pixel-sample-v1/results.json';new_path=BASE/'ppr-expert-contrast-v1/results.json'
    old=json.loads(old_path.read_text());new=json.loads(new_path.read_text());assert old['finished'] and new['finished']
    ids=json.loads((BASE/'ppr-expert-contrast-v1/plan.json').read_text())['ids'];assert len(ids)==4
    files=[dict(r,expert='a') for r in old['files'] if r['role']=='target_a']+new['files']
    allowed={(ROOT/r['local_path']).resolve() for r in files};assert len(allowed)==12
    names=['probe_ppr_reference_identity.py','ppr-reference-identity-protocol.md','triplet_color.py']
    config=dict(ids=ids,source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},expert_a_assets_sha256=digest(old_path),expert_bc_assets_sha256=digest(new_path),main_feature='102 RGB hist/moment values',ridge=1,required_correct=9,required_p=.05)
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    summary=[];primary=[];rows=[]
    for group,key in enumerate(ids):
        for expert,label in zip('abc',range(3),strict=True):
            choices=[r for r in files if r['id']==key and r['expert']==expert];assert len(choices)==1
            row=choices[0];path=ROOT/row['local_path'];assert path.resolve() in allowed and digest(path)==row['sha256']
            with Image.open(path) as im:rgb,_=to_srgb(im);array=np.array(rgb.resize((256,256),Image.Resampling.BILINEAR))
            a,b=transform(array);summary.append(a);primary.append(b);rows.append(dict(id=key,expert=expert,label=label,group=group,sha256=row['sha256']))
    primary=np.array(primary);summary=np.array(summary);labels=np.array([r['label'] for r in rows]);groups=np.array([r['group'] for r in rows])
    folds=operators(primary,groups);main_pred,scores=predict(folds,labels)
    baseline_pred,baseline_scores=predict(operators(summary,groups),labels)
    observed=int((main_pred==labels).sum());permutations=list(itertools.permutations(range(3)));counts=[]
    for choices in itertools.product(permutations,repeat=3):
        assert time.monotonic()-start<60
        perm=np.array([(0,1,2),*choices],dtype=int).reshape(-1)
        pred,_=predict(folds,perm);counts.append(int((pred==perm).sum()))
    assert len(counts)==216
    pvalue=sum(c>=observed for c in counts)/216
    for n,row in enumerate(rows):row.update(predicted_expert='abc'[main_pred[n]],correct=bool(main_pred[n]==labels[n]),secondary_predicted_expert='abc'[baseline_pred[n]],ridge_scores=scores[n].tolist())
    np.savez_compressed(OUT/'features.npz',primary=primary,summary=summary,labels=labels,groups=groups)
    result=dict(finished=True,config_sha256=digest(OUT/'config.json'),rows=rows,main_correct=observed,main_total=12,secondary_correct=int((baseline_pred==labels).sum()),exact_permutation_counts=counts,exact_p=pvalue,reference_identity_signal=bool(observed>=9 and pvalue<=.05),rgb_only=True,original_or_mask_images_read=False,product_candidate_trained=False,software_quality_acceptance=False,seconds=time.monotonic()-start)
    assert all(digest(Path(__file__).with_name(n))==h for n,h in config['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['main_correct','main_total','secondary_correct','exact_p','reference_identity_signal','seconds']}))


if __name__=='__main__':main()
