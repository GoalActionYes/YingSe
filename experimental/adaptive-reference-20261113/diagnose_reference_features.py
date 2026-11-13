"""Controlled-operation reference features; no GT or product promotion."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import pairwise_histograms
from reference_feature_assay import OPERATIONS,transform,appearance_statistics,group_folds,ridge_predict


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path)
    checkpoint=base/'multiref-control-v1/epoch-6.pth';expected='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    if digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c' or digest(checkpoint)!=expected:raise ValueError('Assay inputs changed')
    sources=sorted({r['content'] for r in cohort['groups']['train']});lookup={r['original_path']:r for r in prep['files']}
    if len(sources)!=256 or len({lookup[p]['sha256'] for p in sources})!=256:raise ValueError('Unexpected or duplicate originals')
    folds=group_folds(sources);out=base/'reference-feature-assay-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['diagnose_reference_features.py','reference_feature_assay.py','test_reference_feature_assay.py','reference-feature-assay-protocol.md','train_triplet_pilot.py','pairwise_context.py','training_checks.py','independent_coco.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed training originals, known synthetic operations, grouped linear-probe diagnosis only',cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoint_sha256=expected,source_sha256={p.name:digest(p) for p in frozen.iterdir()},operations=list(OPERATIONS),ridge_lambda=1,folds=folds,rows=[],families={},finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,author=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    features={k:[] for k in ['M','J','A','S']};started=time.perf_counter()
    with torch.inference_mode():
        for i,source in enumerate(sources):
            entry=lookup[source];path=data/entry['prepared_path']
            if digest(path)!=entry['sha256']:raise ValueError('Prepared original changed')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            variants=[transform(rgb,name) for name in OPERATIONS];tensor=torch.from_numpy(np.stack(variants)).permute(0,3,1,2).float().cuda()/255
            color=api['srgb_to_ycbcr'](tensor);m=api['batch_histogram_1D'](color).flatten(1);j=torch.cat([m,pairwise_histograms(color)],1)
            encoded=model.img_projection(api['TF'].resize(tensor,(64,64),antialias=True));mean,std,gram=appearance_statistics(encoded)
            values=dict(M=m,J=j,A=torch.cat([j,mean],1),S=torch.cat([j,mean,std,gram],1))
            for name,value in values.items():
                if not torch.isfinite(value).all():raise ValueError('Nonfinite feature')
                features[name].append(value.cpu().numpy())
            for label,variant in enumerate(variants):
                import hashlib
                report['rows'].append(dict(source=source,original_sha256=entry['sha256'],fold=folds[source],label=label,pixel_sha256=hashlib.sha256(variant.tobytes()).hexdigest()))
            if (i+1)%64==0:print(json.dumps(dict(originals_extracted=i+1)),flush=True)
    torch.cuda.synchronize();report['feature_seconds']=time.perf_counter()-started
    arrays={k:np.concatenate(v) for k,v in features.items()};np.savez_compressed(out/'features.npz',**arrays);report['features_sha256']=digest(out/'features.npz')
    labels=np.array([r['label'] for r in report['rows']]);fold=np.array([r['fold'] for r in report['rows']]);n=len(OPERATIONS)
    for name,values in arrays.items():
        prediction=np.full(len(labels),-1,dtype=np.int64);fold_reports=[]
        for number in [0,1]:
            test=fold==number;train=~test
            if {r['source'] for r,t in zip(report['rows'],train) if t}&{r['source'] for r,t in zip(report['rows'],test) if t}:raise ValueError('Probe photo leakage')
            prediction[test]=ridge_predict(values[train],labels[train],values[test],n)
            fold_reports.append(dict(fold=number,train_rows=int(train.sum()),test_rows=int(test.sum()),accuracy=float((prediction[test]==labels[test]).mean())))
        confusion=np.zeros((n,n),dtype=np.int64);np.add.at(confusion,(labels,prediction),1)
        report['families'][name]=dict(dimensions=values.shape[1],accuracy=float((prediction==labels).mean()),folds=fold_reports,confusion=confusion.tolist(),per_operation_accuracy=(confusion.diagonal()/confusion.sum(1)).tolist(),prediction=prediction.tolist())
        print(json.dumps(dict(family=name,dimensions=values.shape[1],accuracy=report['families'][name]['accuracy'],per_operation_accuracy=report['families'][name]['per_operation_accuracy'])),flush=True)
    if len(report['rows'])!=2816 or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(checkpoint)!=expected or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()) or any(digest(data/lookup[p]['prepared_path'])!=lookup[p]['sha256'] for p in sources):raise ValueError('Frozen assay evidence changed')
    report.update(author_revision=author['revision'],finished=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,summary={k:v['accuracy'] for k,v in report['families'].items()})),flush=True)


if __name__=='__main__':main()
