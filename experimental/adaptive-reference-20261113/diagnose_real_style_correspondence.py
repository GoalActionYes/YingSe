"""Use consumed real training pairs to diagnose fixed reference descriptors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from pairwise_context import pairwise_histograms
from reference_feature_assay import appearance_statistics
from real_style_correspondence import correspondence,gt_differences


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c' or digest(checkpoint)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Changed diagnostic inputs')
    rows=cohort['groups']['train'];lookup={v['original_path']:v for v in prep['files']};groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group_id'],[]).append(i)
    if len(rows)!=1024 or len(groups)!=256 or any(sorted(rows[i]['style_index'] for i in group)!=list(range(4)) or len({rows[i]['content'] for i in group})!=1 for group in groups.values()):raise ValueError('Expected fixed train groups')
    out=base/'real-style-correspondence-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['diagnose_real_style_correspondence.py','real_style_correspondence.py','test_real_style_correspondence.py','real-style-correspondence-protocol.md','reference_feature_assay.py','train_triplet_pilot.py','pairwise_context.py','training_checks.py','independent_coco.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed1024 real train-pair offline GT-assisted correspondence; no independent acceptance',cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoint_sha256=digest(checkpoint),source_sha256={p.name:digest(p) for p in frozen.iterdir()},pixel_records=[],groups=[],families={},finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def read(relative):
        entry=lookup[relative];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Changed parent input')
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return rgb,dict(path=relative,parent_sha256=entry['sha256'],pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    model,api,_=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    arrays={name:[] for name in ['M','J','A','S']};gts=[];hashes=[]
    with torch.inference_mode():
        for offset in range(0,1024,4):
            batch=rows[offset:offset+4];items=[read(row[role]) for row in batch for role in ['reference','gt']];rgb=np.stack([v[0] for v in items]);report['pixel_records'].extend(v[1] for v in items);hashes.extend(v[1]['pixel_sha256'] for v in items);gts.extend(rgb[1::2])
            t=torch.from_numpy(rgb).permute(0,3,1,2).float().cuda()/255;y=api['srgb_to_ycbcr'](t);m=api['batch_histogram_1D'](y).flatten(1);j=torch.cat([m,pairwise_histograms(y)],1);encoded=model.img_projection(api['TF'].resize(t,(64,64),antialias=True));mean,std,gram=appearance_statistics(encoded)
            for name,value in dict(M=m,J=j,A=torch.cat([j,mean],1),S=torch.cat([j,mean,std,gram],1)).items():arrays[name].append(value.cpu().numpy())
            if (offset+4)%256==0:print(json.dumps(dict(train_rows_extracted=offset+4)),flush=True)
    arrays={k:np.concatenate(v) for k,v in arrays.items()};np.savez_compressed(out/'features.npz',**arrays);report['features_sha256']=digest(out/'features.npz')
    for group_id,indices in groups.items():
        indices=sorted(indices,key=lambda i:rows[i]['style_index']);refs=[hashes[i*2] for i in indices];truth=[hashes[i*2+1] for i in indices];differences=gt_differences(np.stack([gts[i] for i in indices]));report['groups'].append(dict(group_id=group_id,indices=[rows[i]['index'] for i in indices],gt_pair_mae=differences,gt_pair_mae_le1=sum(v<=1 for v in differences),positive_exact_duplicates=sum(a==b for a,b in zip(refs,truth)),negative_exact_duplicates=sum(refs[i]==truth[j] for i in range(4) for j in range(4) if i!=j)))
    for name,values in arrays.items():
        values=values.astype(np.float64);values=(values-values.mean(0))/np.maximum(values.std(0),1e-6);scores=[]
        for group_id,indices in groups.items():
            order=sorted(indices,key=lambda i:rows[i]['style_index']);a=values[[i*2 for i in order]];b=values[[i*2+1 for i in order]];cost=np.mean((a[:,None]-b[None])**2,axis=2);scores.append(dict(group_id=group_id,cost=cost.tolist(),**correspondence(cost)))
        report['families'][name]=dict(dimensions=values.shape[1],summary={key:float(np.mean([v[key] for v in scores])) for key in ['row_top1','assignment_fraction_correct','identity_optimum_fraction','diagonal_mean','offdiagonal_mean','margin']},groups=scores);print(json.dumps(dict(family=name,summary=report['families'][name]['summary'])),flush=True)
    differences=[v for g in report['groups'] for v in g['gt_pair_mae']];report['gt_summary']=dict(pairs=len(differences),mean=float(np.mean(differences)),median=float(np.median(differences)),mae_le1=sum(v<=1 for v in differences),positive_exact_duplicates=sum(v['positive_exact_duplicates'] for v in report['groups']),negative_exact_duplicates=sum(v['negative_exact_duplicates'] for v in report['groups']))
    if digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(checkpoint)!=report['checkpoint_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()) or any(digest(data/lookup[v['path']]['prepared_path'])!=v['parent_sha256'] for v in report['pixel_records']):raise ValueError('Frozen diagnostic changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,gt_summary=report['gt_summary'])),flush=True)


if __name__=='__main__':main()
