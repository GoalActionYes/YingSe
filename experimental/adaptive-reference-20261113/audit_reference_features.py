"""Re-extract the frozen assay features and independently recompute probe scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import pairwise_histograms
from reference_feature_assay import OPERATIONS,transform,group_folds


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';study=base/'reference-feature-assay-v1';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    r=load(study/'results.json');before=digest(study/'results.json');checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if not r['finished'] or r['app_promoted'] or r['operations']!=list(OPERATIONS) or digest(checkpoint)!=r['checkpoint_sha256'] or digest(study/'features.npz')!=r['features_sha256'] or digest(data/'cohort.json')!=r['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Frozen inputs changed')
    for name,sha in r['source_sha256'].items():
        if digest(study/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Frozen source changed')
    cohort=load(data/'cohort.json');lookup={v['original_path']:v for v in load(data/'srgb-v1/manifest.json')['files']};sources=sorted({v['content'] for v in cohort['groups']['train']});folds=group_folds(sources)
    if len(sources)!=256 or folds!=r['folds'] or len(r['rows'])!=2816:raise ValueError('Groups changed')
    out=base/'reference-feature-assay-audit-v1';out.mkdir(exist_ok=False);(out/'audit_reference_features.py').write_bytes(Path(__file__).read_bytes())
    arrays=np.load(study/'features.npz');dimensions=dict(M=384,J=1152,A=1280,S=9664)
    if set(arrays.files)!=set(dimensions) or any(arrays[k].shape!=(2816,d) or not np.isfinite(arrays[k]).all() for k,d in dimensions.items()):raise ValueError('Feature cache differs')
    model,api,_=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);maximum={k:0. for k in dimensions}
    with torch.inference_mode():
        for i,source in enumerate(sources):
            entry=lookup[source];path=data/entry['prepared_path']
            if digest(path)!=entry['sha256']:raise ValueError('Parent changed')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            variants=[transform(rgb,name) for name in OPERATIONS];start=i*11
            for j,im in enumerate(variants):
                row=r['rows'][start+j]
                if row!=dict(source=source,original_sha256=entry['sha256'],fold=folds[source],label=j,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()):raise ValueError('Operation order/pixels differ')
            t=torch.from_numpy(np.stack(variants)).permute(0,3,1,2).float().cuda()/255;y=api['srgb_to_ycbcr'](t);m=api['batch_histogram_1D'](y).flatten(1);joint=torch.cat([m,pairwise_histograms(y)],1)
            encoded=model.img_projection(api['TF'].resize(t,(64,64),antialias=True));flat=encoded.flatten(2);mean=flat.mean(-1);std=flat.std(-1,unbiased=False);full=flat@flat.transpose(1,2)/flat.shape[-1];ii,jj=torch.triu_indices(128,128,device=flat.device);gram=full[:,ii,jj]
            values=dict(M=m,J=joint,A=torch.cat([joint,mean],1),S=torch.cat([joint,mean,std,gram],1))
            for name,value in values.items():
                difference=float(np.max(np.abs(value.cpu().numpy()-arrays[name][start:start+11])));maximum[name]=max(maximum[name],difference)
                if difference!=0:raise ValueError('Feature extraction does not reproduce')
            if (i+1)%64==0:print(json.dumps(dict(originals_reproduced=i+1)),flush=True)
    labels=np.array([v['label'] for v in r['rows']]);fold=np.array([v['fold'] for v in r['rows']]);report=dict(source_sha256=digest(Path(__file__)),results_sha256=before,features_sha256=r['features_sha256'],originals=256,rows=2816,maximum_feature_difference=maximum,families={},finished=False,app_promoted=False)
    for name in dimensions:
        values=arrays[name].astype(np.float64);prediction=np.full(2816,-1,dtype=np.int64);residuals=[]
        for number in [0,1]:
            train=fold!=number;test=~train;a=values[train];mean=a.mean(0);std=np.maximum(a.std(0),1e-6);a=(a-mean)/std/np.sqrt(values.shape[1]);b=(values[test]-mean)/std/np.sqrt(values.shape[1]);kernel=a@a.T+np.eye(train.sum());target=np.eye(11)[labels[train]];alpha=np.linalg.solve(kernel,target);residuals.append(float(np.max(np.abs(kernel@alpha-target))));prediction[test]=(b@(a.T@alpha)).argmax(1)
        expected=r['families'][name];confusion=np.zeros((11,11),dtype=np.int64);np.add.at(confusion,(labels,prediction),1)
        if prediction.tolist()!=expected['prediction'] or confusion.tolist()!=expected['confusion'] or float((labels==prediction).mean())!=expected['accuracy'] or (confusion.diagonal()/256).tolist()!=expected['per_operation_accuracy'] or max(residuals)>1e-8:raise ValueError('Probe metrics do not reproduce')
        for item in expected['folds']:
            test=fold==item['fold']
            if item['train_rows']!=1408 or item['test_rows']!=1408 or item['accuracy']!=float((labels[test]==prediction[test]).mean()):raise ValueError('Fold metrics differ')
        report['families'][name]=dict(accuracy=expected['accuracy'],prediction_mismatches=0,linear_solve_maximum_residual=max(residuals));print(json.dumps(dict(family=name,verified=True)),flush=True)
    if digest(study/'results.json')!=before or digest(study/'features.npz')!=r['features_sha256'] or digest(checkpoint)!=r['checkpoint_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in r['source_sha256'].items()):raise ValueError('Audit inputs changed during execution')
    report.update(finished=True,passed=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
