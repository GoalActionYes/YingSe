"""Training-only teacher histograms and unchanged-inference bridge; no optimizer."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from reference_normalization import ReferenceNormalization,histogram_loss


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';coverage=base/'reference-original-coverage-v1';load=lambda p:json.loads(p.read_text())
    registry=load(coverage/'results.json');pairs_path=coverage/'cohort-training-coverage-v2.json';assert digest(pairs_path)==registry['cohorts']['cohort-training-coverage-v2']['detailed_training_report_sha256']
    selected=load(pairs_path)['selected'];assert len(selected)==865
    data=base/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');manifest=data/'srgb-v1/manifest.json';assert digest(manifest)==registry['cohorts'][data.name]['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in load(manifest)['files']};train={r['index']:r for r in cohort['groups']['train']};train_contents={r['content'] for r in train.values()}
    model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True))
    warm=base/'training-coverage-expanded-v2';checkpoint=warm/'epoch-4.pth';assert digest(checkpoint)=='cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval();before=state_hash(model)
    out=base/'reference-normalization-inputs-v1';out.mkdir(exist_ok=False)
    names=['prepare_reference_normalization.py','reference_normalization.py','reference-normalization-protocol.md','appearance_pairwise.py','appearance_replication.py']
    report=dict(scope='Training original-reference labels and dev64 unchanged-inference bridge only; no optimizer',
        registry_sha256=digest(coverage/'results.json'),pair_report_sha256=digest(pairs_path),cohort_sha256=digest(data/'cohort.json'),prepared_manifest_sha256=digest(manifest),
        checkpoint_sha256=digest(checkpoint),model_state_before=before,source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},inputs=[],labels=[],bridge=[],finished=False,training=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def rgb(path):
        e=lookup[path];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    tensor=lambda a:(torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255).cuda()
    paths=sorted({p for _,original,ref in selected for p in [original,ref]});mapping={p:i for i,p in enumerate(paths)};histograms=[]
    assert len(paths)==1267
    for i,path in enumerate(paths):
        a=rgb(path)
        with torch.no_grad():h=api['batch_histogram_1D'](api['srgb_to_ycbcr'](tensor(a))).cpu().numpy()[0]
        assert h.shape==(3,128) and np.isfinite(h).all() and np.allclose(h.sum(-1),1,atol=1e-6)
        histograms.append(h);report['inputs'].append(dict(path=path,parent_sha256=lookup[path]['sha256'],pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    p=out/'histograms.npy';np.save(p,np.stack(histograms));report['histograms_sha256']=digest(p)
    for index,original,reference in selected:
        assert index in train and original in train_contents and train[index]['reference']==reference
        report['labels'].append(dict(index=index,original_teacher=mapping[original],identity_teacher=mapping[reference]))
    torch.manual_seed(20261201);head=ReferenceNormalization().cuda();sample=tensor(rgb(selected[0][2]));features=model.img_projection(api['TF'].resize(sample,(64,64),antialias=True))
    current=api['batch_histogram_1D'](api['srgb_to_ycbcr'](sample));teacher=torch.from_numpy(histograms[mapping[selected[0][1]]][None]).cuda()
    loss=histogram_loss(head(features,current),teacher);loss.backward()
    gradient=sum(float(p.grad.abs().sum()) for p in model.img_projection.parameters() if p.grad is not None);assert gradient>0
    report['shared_encoder_auxiliary_gradient_l1']=gradient;model.zero_grad(set_to_none=True)
    scores={r['index']:r for r in load(warm/'epoch-4/scores.json')['cases']}
    with torch.inference_mode():
        for row in cohort['groups']['validation']:
            # Neither auxiliary head nor teacher original is passed into ordinary inference.
            actual=appearance_forward(model,api,tensor(rgb(row['content'])),tensor(rgb(row['reference'])))[0]
            a=np.rint(actual[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            p=warm/'epoch-4'/f"{row['index']}.png";assert digest(p)==scores[row['index']]['sha256']
            with Image.open(p) as im:assert np.array_equal(a,np.array(im))
            report['bridge'].append(dict(index=row['index'],pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    assert state_hash(model)==before
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(finished=True,model_state_after=before);(out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(finished=True,histograms=len(histograms),labels=len(selected),bridge_exact=len(report['bridge']),shared_encoder_gradient=gradient)),flush=True)


if __name__=='__main__':main()
