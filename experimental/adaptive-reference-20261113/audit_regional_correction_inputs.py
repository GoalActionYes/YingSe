"""Verify all RGB/probability/descriptor lineage and rerun fixed endpoint inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from person_segmenter import PersonSegmenter


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'regional-correction-inputs-v1'
    r=json.loads((out/'results.json').read_text());registry=json.loads((base/'petal-condition-features-v1/results.json').read_text())
    assert r['finished'] and len(r['inputs'])==1365 and len(r['rows'])==1088
    assert digest(base/'petal-condition-features-v1/results.json')==r['registry_sha256']
    assert digest(base/'cohort-multiref-v1/cohort.json')==r['cohort_sha256']
    assert digest(base/'cohort-multiref-v1/srgb-v1/manifest.json')==r['prepared_manifest_sha256']
    assert r['model_state_before']==r['model_state_after']
    for n,h in r['source_sha256'].items():assert digest(out/'frozen-source'/n)==h and digest(Path(__file__).with_name(n))==h
    assert digest(out/'descriptors.npz')==r['descriptors_sha256']
    with np.load(out/'descriptors.npz',allow_pickle=False) as a:features=a['regional']
    assert features.shape==(1365,1154) and features.dtype==np.float32 and np.isfinite(features).all()
    matrix=np.array([[.2126,.7152,.0722],[-.1146,-.3854,.5],[.5,-.4542,-.0458]],np.float32)
    endpoints=[];maximum=0.
    for item,prior in zip(r['inputs'],registry['unique_inputs']):
        assert item['index']==prior['index'] and item['pixel_sha256']==prior['pixel_sha256'] and item['path']==prior['path']
        p=ROOT/item['path'];assert digest(p)==item['parent_sha256']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['pixel_sha256']
        assert digest(out/item['mask_path'])==item['mask_sha256'];prob=np.load(out/item['mask_path'],allow_pickle=False)
        assert prob.dtype==np.float32 and prob.shape==(256,256) and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
        assert float(prob.mean())==item['soft_person_fraction'] and float((prob>.5).mean())==item['hard_person_fraction']
        x=rgb.astype(np.float32)/np.float32(255);ycc=x@matrix.T
        ycc[...,0]=ycc[...,0].clip(0,1);ycc[...,1:]=ycc[...,1:].clip(-.5,.5)+np.float32(.5)
        bins=np.floor(ycc*128).clip(0,127).astype(int)
        expected=[]
        for w in [np.ones(prob.shape,np.float32),prob,1-prob]:
            total=float(w.astype(np.float64).sum());h=np.zeros((3,128),np.float32)
            if total:
                for channel in range(3):
                    np.add.at(h[channel],bins[...,channel].ravel(),np.zeros(1,dtype=np.float32)[0])
                    sums=np.zeros(128,np.float64);np.add.at(sums,bins[...,channel].ravel(),w.ravel().astype(np.float64));h[channel]=sums/total
            expected.extend(h.ravel())
        expected.extend([prob.mean(),(prob>.5).mean()]);err=float(np.abs(np.array(expected,np.float32)-features[item['index']]).max());maximum=max(maximum,err);assert err<=1e-7
        if item['index'] in (0,1364):endpoints.append((item,rgb,prob))
    assert r['rows']==registry['rows']
    lookup={x['index']:x for x in r['inputs']};support={}
    for split in ['train','validation']:
        rows=[x for x in r['rows'] if x['split']==split]
        supported=0
        for row in rows:
            for role in ['content','reference']:
                source=lookup[row['roles'][role]['feature_index']];assert source['pixel_sha256']==row['roles'][role]['pixel_sha256']
            supported+=all(.02<=lookup[row['roles'][role]['feature_index']]['soft_person_fraction']<=.98 and .02<=lookup[row['roles'][role]['feature_index']]['hard_person_fraction']<=.98 for role in ['content','reference'])
        support[split]=dict(rows=len(rows),both_regions_supported=supported)
    assert support==r['support']
    engine=PersonSegmenter();repeats=[]
    for item,rgb,prob in endpoints:
        actual,_=engine.predict(rgb);difference=float(np.abs(actual-prob).max());assert difference==0
        repeats.append(dict(index=item['index'],maximum_probability_difference=difference))
    evidence=dict(scope=r['scope'],results_sha256=digest(out/'results.json'),source_sha256=r['source_sha256'],audit_script_sha256=digest(Path(__file__)),
        inputs=1365,rows=1088,descriptors_sha256=r['descriptors_sha256'],maximum_descriptor_recalculation_difference=maximum,
        support=support,segmentation_endpoint_repeats=repeats,shared_train_development_input_pixels=r['shared_train_development_input_pixels'],
        segmentation_state_before=r['model_state_before'],segmentation_state_after=r['model_state_after'],
        exact_fallback_development_improvement_upper_bound=28,existing_required_improvement_count=40,
        original_exact_fallback_design_can_pass=False,training_started=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-regional-inputs-evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
