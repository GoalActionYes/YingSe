"""Truth-assisted offline interpolation test; never an automatic grading score."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from triplet_color import to_srgb

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-spatial-information-v1'
SAMPLES=BASE/'ppr-pixel-sample-v1'
EXPERTS=BASE/'ppr-expert-contrast-v1'
METHODS=['global3','global5','human3','position3','permuted3']


def basis(rgb,degree):
    x=np.asarray(rgb,dtype=np.float64).reshape(-1,3)/127.5-1
    polynomials=[np.polynomial.legendre.legvander(x[:,c],degree) for c in range(3)]
    exponents=[e for e in itertools.product(range(degree+1),repeat=3) if sum(e)<=degree]
    exponents.sort(key=lambda e:(sum(e),e))
    return np.stack([polynomials[0][:,a]*polynomials[1][:,b]*polynomials[2][:,c] for a,b,c in exponents],axis=1)


def metrics(prediction,target,alpha):
    error=np.abs(prediction.astype(np.float64)-target.astype(np.float64))
    return dict(all_mae=float(error.mean()),human_mae=float((error*alpha[:,:,None]).sum()/(alpha.sum()*3)),background_mae=float((error*(1-alpha)[:,:,None]).sum()/((1-alpha).sum()*3)))


def signal(rows):
    groups=[]
    for key in dict.fromkeys(r['id'] for r in rows):
        values=[r for r in rows if r['id']==key];scores={m:float(np.mean([r['metrics'][m]['all_mae'] for r in values])) for m in METHODS}
        gains={m:dict(relative=(scores[m]-scores['human3'])/scores[m],absolute=scores[m]-scores['human3']) for m in ['global5','position3','permuted3']}
        regions_ok=all(np.mean([r['metrics']['human3'][region]-r['metrics']['global5'][region] for r in values])<=.1 for region in ['human_mae','background_mae'])
        passed=regions_ok and all(v['relative']>=.05 and v['absolute']>=.25 for v in gains.values())
        groups.append(dict(id=key,mean_all_mae=scores,gains=gains,regions_ok=bool(regions_ok),signal_passed=bool(passed)))
    return dict(groups=groups,groups_passed=sum(g['signal_passed'] for g in groups),required_groups=3,necessary_spatial_signal=bool(sum(g['signal_passed'] for g in groups)>=3),product_acceptance=False)


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    p=json.loads((SAMPLES/'plan.json').read_text());original=json.loads((SAMPLES/'results.json').read_text());more=json.loads((EXPERTS/'results.json').read_text())
    assert original['finished'] and more['finished']
    identity=BASE/'ppr-source-pixel-link-v1/results.json';known=json.loads(identity.read_text())
    assert known['finished'] and all(r['same_file_bytes'] for r in known['rows'])
    names=['diagnose_ppr_spatial_information.py','ppr-spatial-information-protocol.md','triplet_color.py']
    config=dict(ids=p['ids'],source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},source_identity_sha256=digest(identity),original_results_sha256=digest(SAMPLES/'results.json'),expert_results_sha256=digest(EXPERTS/'results.json'),ridge=1e-6,block=8,threads=2)
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    settings=ROOT/'.tools/PPR10K-audit/xmp_target_a.zip'
    assert digest(settings)=='b02e46d25ef0aeee4c6c6a1bc5dc517066bf8b009696a70d7db1f48083d04a7a'
    local_flags=[]
    with zipfile.ZipFile(settings) as z:
        for key in p['ids']:
            candidates=[n for n in z.namelist() if Path(n).stem==key];assert len(candidates)==1
            xml=ET.fromstring(z.read(candidates[0]));fields=set()
            for node in xml.iter():
                if node.tag.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):fields.add(node.tag.split('}',1)[1])
                fields.update(k.split('}',1)[1] for k in node.attrib if k.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'))
            local_flags.append(dict(id=key,local_structure_fields=sorted(fields&{'PaintBasedCorrections','CorrectionMasks','GradientBasedCorrections','CircularGradientBasedCorrections'})))
    all_targets=[dict(r,expert='a') for r in original['files'] if r['role']=='target_a']+more['files']
    yy,xx=np.indices((256,256));fold=((xx//8+yy//8)%2).reshape(-1)
    records=[];fits=[];predictions={};input_arrays={}
    for number,key in enumerate(p['ids']):
        assert time.monotonic()-start<180
        def read(row,mask=False):
            path=ROOT/row['local_path'];assert digest(path)==row['sha256']
            with Image.open(path) as im:
                if mask:return np.array(im.convert('L').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float64)/255
                rgb,_=to_srgb(im)
                return np.array(rgb.resize((256,256),Image.Resampling.BILINEAR))
        source=read(next(r for r in original['files'] if r['role']=='source' and r['id']==key))
        alpha=read(next(r for r in original['files'] if r['role']=='mask' and r['id']==key),True)
        targets=[read(next(r for r in all_targets if r['expert']==e and r['id']==key)) for e in 'abc']
        y=np.concatenate([t.reshape(-1,3)/255 for t in targets],axis=1)
        cubic=basis(source,3);quintic=basis(source,5)
        assert cubic.shape[1]==20 and quintic.shape[1]==56
        shuffled=np.random.default_rng(20261118+number).permutation(alpha.reshape(-1))
        features={'global3':cubic,'global5':quintic,'human3':np.c_[cubic,cubic*alpha.reshape(-1,1)],'position3':np.c_[cubic,cubic*(xx.reshape(-1,1)/255)],'permuted3':np.c_[cubic,cubic*shuffled[:,None]]}
        for method,x in features.items():
            predicted=np.empty_like(y)
            for parity in [0,1]:
                train=fold==parity;test=~train
                gram=x[train].T@x[train]/train.sum();rhs=x[train].T@y[train]/train.sum()
                penalty=np.eye(x.shape[1])*1e-6;penalty[0,0]=0
                if x.shape[1]==40:penalty[20,20]=1e-6
                coefficients=np.linalg.solve(gram+penalty,rhs)
                assert np.isfinite(coefficients).all()
                predicted[test]=x[test]@coefficients
                fits.append(dict(id=key,method=method,fold=parity,features=x.shape[1],regularized_condition_number=float(np.linalg.cond(gram+penalty))))
            result=np.rint(np.clip(predicted,0,1)*255).astype(np.uint8).reshape(256,256,9)
            predictions[key+'_'+method]=result
        input_arrays[key+'_source']=source;input_arrays[key+'_alpha']=alpha
        for n,expert in enumerate('abc'):
            rows={m:metrics(predictions[key+'_'+m][:,:,n*3:n*3+3],targets[n],alpha) for m in METHODS}
            records.append(dict(id=key,expert=expert,metrics=rows))
            input_arrays[key+'_'+expert]=targets[n]
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    np.savez_compressed(OUT/'inputs.npz',**input_arrays)
    decision=signal(records)
    report=dict(finished=True,config_sha256=digest(OUT/'config.json'),source_sha256=config['source_sha256'],truth_assisted=True,software_inference=False,rows=records,fits=fits,expert_a_local_metadata=local_flags,decision=decision,seconds=time.monotonic()-start,training_product_candidate=False,product_promoted=False)
    assert all(digest(Path(__file__).with_name(n))==h for n,h in config['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(decision=decision,seconds=report['seconds'],local_metadata=local_flags)))


if __name__=='__main__':main()
