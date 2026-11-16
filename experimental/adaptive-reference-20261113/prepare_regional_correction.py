"""Input-only automatic region cache; no GT read, optimizer or color-model update."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from person_segmenter import PersonSegmenter,CHECKPOINT_SHA
from regional_descriptors import describe


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';registry_path=base/'petal-condition-features-v1/results.json'
    registry=json.loads(registry_path.read_text());assert registry['finished'] and not registry['gt_features_reused']
    data=base/'cohort-multiref-v1';assert digest(data/'cohort.json')==registry['cohort_sha256']
    assert digest(data/'srgb-v1/manifest.json')==registry['prepared_manifest_sha256']
    checkpoint=ROOT/'.tools/torch-cache/segmentation/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
    assert checkpoint.is_file() and digest(checkpoint)==CHECKPOINT_SHA,'No downloads permitted'
    out=base/'regional-correction-inputs-v1';out.mkdir(exist_ok=False);(out/'masks').mkdir();(out/'frozen-source').mkdir()
    names=['regional-correction-preparation-protocol.md','prepare_regional_correction.py','regional_descriptors.py','test_regional_descriptors.py','person_segmenter.py']
    for n in names:(out/'frozen-source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    model=PersonSegmenter();before=state_hash(model.model)
    report=dict(scope='Consumed training1024/development64 RGB-only automatic regional preparation; no trained correction yet',
        registry_sha256=digest(registry_path),cohort_sha256=registry['cohort_sha256'],prepared_manifest_sha256=registry['prepared_manifest_sha256'],
        source_sha256={n:digest(out/'frozen-source'/n) for n in names},provenance=model.provenance,
        tf32=False,model_state_before=before,rows=registry['rows'],inputs=[],training_started=False,finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    inputs=registry['unique_inputs'];descriptors=np.empty((len(inputs),1154),np.float32)
    for item in inputs:
        p=ROOT/item['path'];assert digest(p)==item['parent_sha256']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['pixel_sha256']
        probability,timing=model.predict(rgb)
        assert probability.shape==(256,256) and probability.dtype==np.float32 and np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
        index=item['index'];path=out/'masks'/f'{index:04d}.npy';np.save(path,probability,allow_pickle=False)
        descriptors[index]=describe(rgb,probability)
        report['inputs'].append(dict(index=index,path=item['path'],parent_sha256=item['parent_sha256'],pixel_sha256=item['pixel_sha256'],
            mask_path=str(path.relative_to(out)),mask_sha256=digest(path),soft_person_fraction=float(probability.mean()),
            hard_person_fraction=float((probability>.5).mean()),seconds=timing['seconds']))
        if len(report['inputs'])%100==0:
            (out/'progress.json').write_text(json.dumps(dict(prepared=len(report['inputs']),total=len(inputs))))
            print(json.dumps(dict(prepared=len(report['inputs']),total=len(inputs))),flush=True)
    np.savez_compressed(out/'descriptors.npz',regional=descriptors);report['descriptors_sha256']=digest(out/'descriptors.npz')
    report['model_state_after']=state_hash(model.model);assert report['model_state_after']==before
    lookup={x['index']:x for x in report['inputs']};support={}
    for split in ['train','validation']:
        rows=[x for x in report['rows'] if x['split']==split];assert len(rows)==(1024 if split=='train' else 64)
        ok=lambda x:.02<=x['soft_person_fraction']<=.98 and .02<=x['hard_person_fraction']<=.98
        support[split]=dict(rows=len(rows),both_regions_supported=sum(all(ok(lookup[x['roles'][role]['feature_index']]) for role in ['content','reference']) for x in rows))
    report['support']=support
    pixels={split:{x['roles'][role]['pixel_sha256'] for x in report['rows'] if x['split']==split for role in ['content','reference']} for split in ['train','validation']}
    report['shared_train_development_input_pixels']=len(pixels['train']&pixels['validation'])
    assert digest(registry_path)==report['registry_sha256'] and all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(finished=True,inputs=len(inputs),support=support,shared_pixels=report['shared_train_development_input_pixels'],training_started=False)),flush=True)


if __name__=='__main__':main()
