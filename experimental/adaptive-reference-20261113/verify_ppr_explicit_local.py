"""SVD refit and independent scoring of the offline, truth-assisted diagnosis."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import itertools
import json
import time
import numpy as np
from independent_coco import ROOT,digest

OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-explicit-local-fit-v1'


def features(rgb,degree):
    x=rgb.reshape(-1,3).astype(np.float64)/127.5-1
    values=[np.ones_like(x),x]
    for n in range(2,degree+1):values.append(((2*n-1)*x*values[n-1]-(n-1)*values[n-2])/n)
    powers=sorted((p for p in itertools.product(range(degree+1),repeat=3) if sum(p)<=degree),key=lambda p:(sum(p),p))
    return np.column_stack([values[a][:,0]*values[b][:,1]*values[c][:,2] for a,b,c in powers])


def main():
    start=time.monotonic();d=json.loads((OUT/'results.json').read_text());c=json.loads((OUT/'config.json').read_text())
    assert d['finished'] and d['truth_assisted'] and not d['software_inference'] and not d['product_promoted']
    assert digest(OUT/'config.json')==d['config_sha256']
    for n,h in c['source_sha256'].items():assert digest(OUT/'frozen-source'/n)==h
    inputs=np.load(OUT/'inputs.npz');predictions=np.load(OUT/'predictions.npz')
    yy,xx=np.mgrid[:256,:256];fold=((xx//8+yy//8)%2).reshape(-1)
    max_pixel_delta=0;max_mean_delta=0.;max_score_delta=0.;refits=0
    recalculated={}
    for number,key in enumerate(c['ids']):
        assert time.monotonic()-start<180
        rgb=inputs[key+'_source'];alpha=inputs[key+'_alpha'];a=alpha.reshape(-1,1)
        x3=features(rgb,3);x5=features(rgb,5)
        shuffled=np.random.default_rng(20261118+number).permutation(a[:,0])
        matrices={'global3':x3,'global5':x5,'human3':np.concatenate([x3,x3*a],axis=1),'position3':np.concatenate([x3,x3*(xx.reshape(-1,1)/255)],axis=1),'permuted3':np.concatenate([x3,x3*shuffled[:,None]],axis=1)}
        y=np.column_stack([inputs[key+'_'+e].reshape(-1,3) for e in 'a']).astype(np.float64)/255
        for method,x in matrices.items():
            full=np.empty_like(y)
            penalty=np.eye(x.shape[1])*np.sqrt(c['ridge']);penalty[0,0]=0
            for parity in [0,1]:
                train=fold==parity;test=~train
                design=np.vstack([x[train]/np.sqrt(train.sum()),penalty])
                rhs=np.vstack([y[train]/np.sqrt(train.sum()),np.zeros((x.shape[1],3))])
                coef=np.linalg.lstsq(design,rhs,rcond=None)[0]
                full[test]=np.matmul(x[test],coef);refits+=1
            image=np.rint(255*np.clip(full,0,1)).astype(np.uint8).reshape(256,256,3)
            saved=predictions[key+'_'+method]
            delta=np.abs(image.astype(np.int16)-saved.astype(np.int16))
            max_pixel_delta=max(max_pixel_delta,int(delta.max()));max_mean_delta=max(max_mean_delta,float(delta.mean()))
            for n,expert in enumerate('a'):
                target=inputs[key+'_'+expert].astype(np.int32)
                def scores(value):
                    error=np.abs(value[:,:,n*3:n*3+3].astype(np.int32)-target)
                    return {'all_mae':float(error.sum()/error.size),'human_mae':float(np.einsum('hwc,hw->',error,alpha)/(alpha.sum()*3)),'background_mae':float(np.einsum('hwc,hw->',error,1-alpha)/((1-alpha).sum()*3))}
                metrics=scores(saved);svd=scores(image)
                record=next(r for r in d['rows'] if r['id']==key and r['expert']==expert)['metrics'][method]
                assert max(abs(metrics[k]-record[k]) for k in record)<1e-9
                max_score_delta=max(max_score_delta,*[abs(metrics[k]-svd[k]) for k in record])
                recalculated[key,expert,method]=svd
    groups=[]
    for key in c['ids']:
        all_mae={m:np.mean([recalculated[key,e,m]['all_mae'] for e in 'a']) for m in matrices}
        region_ok=all(np.mean([recalculated[key,e,'human3'][r]-recalculated[key,e,'global5'][r] for e in 'a'])<=.1 for r in ['human_mae','background_mae'])
        good=region_ok and all((all_mae[m]-all_mae['human3'])/all_mae[m]>=.05 and all_mae[m]-all_mae['human3']>=.25 for m in ['global5','position3','permuted3'])
        groups.append(bool(good))
    assert sum(groups)==d['decision']['groups_passed'] and (sum(groups)>=3)==d['decision']['necessary_spatial_signal']
    assert max_pixel_delta<=1 and max_mean_delta<=1e-4 and max_score_delta<=1e-4
    result=dict(finished=True,results_sha256=digest(OUT/'results.json'),driver_sha256=digest(__file__),svd_refits=refits,all_target_method_scores_recomputed=15,max_quantized_pixel_difference=max_pixel_delta,max_quantized_mean_difference=max_mean_delta,max_score_difference=max_score_delta,signal_decision_unchanged=True,necessary_spatial_signal=d['decision']['necessary_spatial_signal'],seconds=time.monotonic()-start,software_acceptance=False)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
