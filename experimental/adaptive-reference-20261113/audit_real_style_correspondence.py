"""Recompute offline real-pair costs and GT diagnostics from retained inputs."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';study=base/'real-style-correspondence-v1';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(study/'results.json');results_sha=digest(study/'results.json')
    checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if not r['finished'] or r['app_promoted'] or digest(checkpoint)!=r['checkpoint_sha256'] or digest(study/'features.npz')!=r['features_sha256'] or digest(data/'cohort.json')!=r['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Changed diagnostic snapshot')
    for name,sha in r['source_sha256'].items():
        if digest(study/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed diagnostic implementation')
    rows=load(data/'cohort.json')['groups']['train'];lookup={v['original_path']:v for v in load(data/'srgb-v1/manifest.json')['files']};groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group_id'],[]).append(i)
    if len(groups)!=256 or len(rows)!=1024 or len(r['pixel_records'])!=2048:raise ValueError('Incomplete diagnostic')
    out=base/'real-style-correspondence-audit-v1';out.mkdir(exist_ok=False);(out/'audit_real_style_correspondence.py').write_bytes(Path(__file__).read_bytes())
    gts=[];hashes=[];count=0
    for row in rows:
        for role in ['reference','gt']:
            item=lookup[row[role]];path=data/item['prepared_path']
            if digest(path)!=item['sha256']:raise ValueError('Changed parent')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            pixel_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            if r['pixel_records'][count]!=dict(path=row[role],parent_sha256=item['sha256'],pixel_sha256=pixel_sha):raise ValueError('Wrong feature-row pixel correspondence')
            hashes.append(pixel_sha);count+=1
            if role=='gt':gts.append(rgb)
    all_differences=[];positive=negative=0
    for saved,(group_id,indices) in zip(r['groups'],groups.items()):
        order=sorted(indices,key=lambda i:rows[i]['style_index']);values=[float(np.abs(gts[i].astype(float)-gts[j].astype(float)).mean()) for i,j in itertools.combinations(order,2)];p=sum(hashes[2*i]==hashes[2*i+1] for i in order);n=sum(hashes[2*i]==hashes[2*j+1] for i in order for j in order if i!=j)
        expected=dict(group_id=group_id,indices=[rows[i]['index'] for i in order],gt_pair_mae=values,gt_pair_mae_le1=sum(v<=1 for v in values),positive_exact_duplicates=p,negative_exact_duplicates=n)
        if saved!=expected:raise ValueError('GT pair diagnostics differ')
        positive+=p;negative+=n;all_differences.extend(values)
    summary=dict(pairs=1536,mean=float(np.mean(all_differences)),median=float(np.median(all_differences)),mae_le1=sum(v<=1 for v in all_differences),positive_exact_duplicates=positive,negative_exact_duplicates=negative)
    if summary!=r['gt_summary']:raise ValueError('GT summary differs')
    with np.load(study/'features.npz') as cache:arrays={k:cache[k] for k in ['M','J','A','S']}
    report=dict(results_sha256=results_sha,source_sha256=digest(Path(__file__)),features_sha256=r['features_sha256'],pixel_rows_verified=count,gt_summary=summary,families={},finished=False,app_promoted=False)
    permutations=list(itertools.permutations(range(4)))
    for name,values in arrays.items():
        if values.shape!=(2048,r['families'][name]['dimensions']) or not np.isfinite(values).all():raise ValueError('Invalid feature cache')
        x=values.astype(float);x=(x-x.mean(0))/np.maximum(x.std(0),1e-6);metrics=[]
        for saved,(group_id,indices) in zip(r['families'][name]['groups'],groups.items()):
            order=sorted(indices,key=lambda i:rows[i]['style_index']);a=x[[i*2 for i in order]];b=x[[i*2+1 for i in order]];cost=np.mean((a[:,None]-b[None])**2,axis=2)
            if saved['group_id']!=group_id or not np.array_equal(cost,np.asarray(saved['cost'])):raise ValueError('Cost matrices differ')
            row_scores=[]
            for i in range(4):
                tied=[j for j in range(4) if np.isclose(cost[i,j],min(cost[i]),rtol=1e-10,atol=1e-12)];row_scores.append((i in tied)/len(tied))
            sums=[sum(cost[i,p[i]] for i in range(4)) for p in permutations];best=[j for j,v in enumerate(sums) if np.isclose(v,min(sums),rtol=1e-10,atol=1e-12)];diagonal=float(np.trace(cost)/4);off=float(cost[~np.eye(4,dtype=bool)].mean())
            actual=dict(row_top1=float(np.mean(row_scores)),assignment_fraction_correct=float(np.mean([sum(i==p for i,p in enumerate(permutations[j]))/4 for j in best])),identity_optimum_fraction=float((0 in best)/len(best)),optimal_assignments=len(best),diagonal_mean=diagonal,offdiagonal_mean=off,margin=off-diagonal)
            if any(saved[k]!=v for k,v in actual.items()):raise ValueError('Tie-aware metric differs')
            metrics.append(actual)
        aggregate={k:float(np.mean([v[k] for v in metrics])) for k in r['families'][name]['summary']}
        if aggregate!=r['families'][name]['summary']:raise ValueError('Aggregate differs')
        report['families'][name]=dict(groups=256,maximum_cost_difference=0,summary=aggregate)
    if digest(study/'results.json')!=results_sha or digest(study/'features.npz')!=r['features_sha256'] or digest(checkpoint)!=r['checkpoint_sha256']:raise ValueError('Snapshot changed during audit')
    report.update(finished=True,passed=True);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
