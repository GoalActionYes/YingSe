"""Frozen provider recipes on consumed train correspondences, no transfer learning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from frozen_grade_encoder import load_frozen_grade_encoder
from real_style_correspondence import correspondence


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c' or digest(checkpoint)!='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9':raise ValueError('Changed diagnostic inputs')
    rows=cohort['groups']['train'];lookup={v['original_path']:v for v in prep['files']};groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group_id'],[]).append(i)
    if len(rows)!=1024 or len(groups)!=256 or any(sorted(rows[i]['style_index'] for i in g)!=list(range(4)) or len({rows[i]['content'] for i in g})!=1 for g in groups.values()):raise ValueError('Unexpected train grouping')
    bridge_path=base/'frozen-grade-bridge-v1/results.json';bridge=load(bridge_path)
    if not bridge['finished'] or not bridge['passed'] or bridge['state_tensors_identical']!=316 or any(v['maximum_float_difference']!=0 for v in bridge['cases']):raise ValueError('Missing native grade bridge')
    for name,sha in bridge['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('Native bridge source changed')
    out=base/'frozen-grade-correspondence-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    names=set(bridge['source_sha256'])|{'diagnose_frozen_grade.py','train_triplet_pilot.py','real_style_correspondence.py'}
    for name in sorted(names):(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    grade=load_frozen_grade_encoder('cuda')
    if grade.provenance!=bridge['provenance']:raise ValueError('Grade provider provenance differs from native bridge')
    before={k:v.detach().cpu().clone() for k,v in grade.encoder.state_dict().items()};model,api,author=author_model();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    report=dict(scope='Consumed1024 training-pair frozen-provider correspondence, possible source/preset overlap, no independent quality claim',source_sha256={p.name:digest(p) for p in frozen.iterdir()},bridge_results_sha256=digest(bridge_path),grade_provenance=grade.provenance,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoint_sha256=digest(checkpoint),pixel_records=[],families={},finished=False,app_promoted=False,actual_independent_eligibility=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');features={name:[] for name in ['H128','G64']}
    def read(relative):
        item=lookup[relative];path=data/item['prepared_path']
        if digest(path)!=item['sha256']:raise ValueError('Changed train pixel input')
        with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return rgb,dict(path=relative,parent_sha256=item['sha256'],pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    with torch.no_grad():
        for offset in range(0,1024,4):
            items=[read(row[k]) for row in rows[offset:offset+4] for k in ['reference','gt']];report['pixel_records'].extend(v[1] for v in items);rgb=torch.from_numpy(np.stack([v[0] for v in items])).permute(0,3,1,2).float().cuda()/255
            h=F.normalize(model.img_projection(api['TF'].resize(rgb,(64,64),antialias=True)).mean((-2,-1)),dim=1,eps=1e-8);g=grade(rgb)
            if not torch.isfinite(h).all() or not torch.isfinite(g).all():raise ValueError('Nonfinite descriptors')
            features['H128'].append(h.cpu().numpy());features['G64'].append(g.cpu().numpy())
            if (offset+4)%256==0:print(json.dumps(dict(train_rows_extracted=offset+4)),flush=True)
    arrays={k:np.concatenate(v) for k,v in features.items()};np.savez_compressed(out/'features.npz',**arrays);report['features_sha256']=digest(out/'features.npz')
    for name,value in arrays.items():
        value=value.astype(np.float64);norm=np.linalg.norm(value,axis=1,keepdims=True);normalized=value/np.maximum(norm,1e-12);results=[]
        for group_id,indices in groups.items():
            order=sorted(indices,key=lambda i:rows[i]['style_index']);a=normalized[[i*2 for i in order]];b=normalized[[i*2+1 for i in order]];cost=np.clip(1-a@b.T,0,2);results.append(dict(group_id=group_id,cost=cost.tolist(),**correspondence(cost)))
        summary={key:float(np.mean([v[key] for v in results])) for key in ['row_top1','assignment_fraction_correct','identity_optimum_fraction','diagonal_mean','offdiagonal_mean','margin']}
        report['families'][name]=dict(dimensions=value.shape[1],norm_summary=dict(minimum=float(norm.min()),median=float(np.median(norm)),maximum=float(norm.max()),zero_vectors=int((norm<=1e-12).sum())),summary=summary,groups=results);print(json.dumps(dict(family=name,summary=summary)),flush=True)
    h,g=[report['families'][name]['summary'] for name in ['H128','G64']];report['research_value_checks']=dict(grade_top1_at_least_half=g['row_top1']>=.5,grade_top1_gain_at_least_five_points=g['row_top1']>=h['row_top1']+.05,grade_assignment_no_worse=g['assignment_fraction_correct']>=h['assignment_fraction_correct']);report['may_design_training_comparison']=all(report['research_value_checks'].values())
    if grade.training or any(p.requires_grad for p in grade.parameters()) or any(not torch.equal(v,grade.encoder.state_dict()[k].cpu()) for k,v in before.items()):raise ValueError('Frozen grade encoder changed')
    if digest(bridge_path)!=report['bridge_results_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(checkpoint)!=report['checkpoint_sha256'] or digest(ROOT/'.tools/CanonCGT/pretrained/E2E.pth')!=grade.provenance['checkpoint_sha256'] or any(digest(ROOT/'.tools/CanonCGT'/n)!=sha for n,sha in grade.provenance['upstream_source_sha256'].items()) or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()) or any(digest(data/lookup[v['path']]['prepared_path'])!=v['parent_sha256'] for v in report['pixel_records']):raise ValueError('Frozen diagnostic evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,may_design_training_comparison=report['may_design_training_comparison'],research_value_checks=report['research_value_checks'],actual_independent_eligibility=False)),flush=True)


if __name__=='__main__':main()
