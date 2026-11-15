"""Prepare content/reference-only frozen descriptors; no optimizer, no GT features."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from petal_model import PetalModel
from petal_encoder import encode


def main():
    torch.set_num_threads(4)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';prior_dir=base/'petal-encoder-correspondence-v1'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    prior_path=prior_dir/'results.json';prior=load(prior_path);cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    if not prior['finished'] or digest(cohort_path)!=prior['cohort_sha256'] or digest(prep_path)!=prior['prepared_manifest_sha256'] or digest(prior_dir/'features.npz')!=prior['features_sha256']:raise ValueError('Historical evidence changed')
    cohort=load(cohort_path);lookup={v['original_path']:v for v in load(prep_path)['files']}
    with np.load(prior_dir/'features.npz') as cache:old=cache['PETAL1024'].copy()
    reused={}
    for i,row in enumerate(cohort['groups']['train']):
        record=prior['pixel_records'][2*i]
        if record['path']!=row['reference']:raise ValueError('Historical reference order differs')
        key=record['pixel_sha256'];value=old[2*i]
        if key in reused and not np.array_equal(reused[key],value):raise ValueError('Duplicate pixel features disagree')
        reused[key]=value
    unique={};rows=[]
    def read(path):
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for split,count in [('train',1024),('validation',64)]:
        if len(cohort['groups'][split])!=count:raise ValueError('Wrong split size')
        for row in cohort['groups'][split]:
            mapping={}
            for role in ['content','reference']:
                relative=row[role];entry=lookup[relative];path=data/entry['prepared_path']
                if digest(path)!=entry['sha256']:raise ValueError('Prepared pixels changed')
                rgb=read(path);key=hashlib.sha256(rgb.tobytes()).hexdigest()
                item=unique.setdefault(key,dict(index=len(unique),pixel_sha256=key,path=str(path.relative_to(ROOT)),parent_sha256=entry['sha256'],reused_reference_only=key in reused))
                mapping[role]=dict(original_path=relative,feature_index=item['index'],pixel_sha256=key,parent_sha256=entry['sha256'])
            rows.append(dict(split=split,index=row['index'],group_id=row['group_id'],style_index=row['style_index'],roles=mapping))
    out=base/'petal-condition-features-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=['prepare_petal_condition.py','petal-condition-protocol.md','petal_model.py','petal_encoder.py','independent_coco.py']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Consumed train1024/development64 content/reference-only frozen feature preparation',source_sha256={n:digest(frozen/n) for n in names},prior_results_sha256=digest(prior_path),prior_features_sha256=prior['features_sha256'],cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),rows=rows,unique_inputs=list(unique.values()),training_started=False,gt_features_reused=False,finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model=PetalModel();report['provenance']=model.provenance;report['model_state_before']=model.state_digest()
    if report['model_state_before']!=prior['model_state_after_sha256']:raise ValueError('Public model state differs from diagnostic')
    values=np.empty((len(unique),1024),np.float32)
    for key,item in unique.items():
        if key in reused:values[item['index']]=reused[key]
    pending=[v for v in unique.values() if not v['reused_reference_only']]
    print(json.dumps(dict(unique=len(unique),reused=len(unique)-len(pending),new=len(pending))),flush=True)
    for offset in range(0,len(pending),2):
        batch=pending[offset:offset+2];images=[]
        for item in batch:
            path=ROOT/item['path'];rgb=read(path)
            if digest(path)!=item['parent_sha256'] or hashlib.sha256(rgb.tobytes()).hexdigest()!=item['pixel_sha256']:raise ValueError('New input changed')
            images.append(rgb)
        if len(images)==1:images.append(images[0])
        features=encode(model,images).cpu().numpy()
        for item,value in zip(batch,features):values[item['index']]=value
        if (offset+2)%64==0:print(json.dumps(dict(new_extracted=min(offset+2,len(pending)),total=len(pending))),flush=True)
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,rtol=1e-5,atol=1e-6):raise ValueError('Invalid style features')
    np.savez_compressed(out/'features.npz',PETAL1024=values);report['features_sha256']=digest(out/'features.npz')
    report['model_state_after']=model.state_digest()
    if report['model_state_before']!=report['model_state_after']:raise ValueError('Frozen model changed')
    if digest(prior_path)!=report['prior_results_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,unique=len(unique),new=len(pending),training_started=False)),flush=True)


if __name__=='__main__':main()
