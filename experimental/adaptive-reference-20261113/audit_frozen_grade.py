"""Audit provider diagnosis from input pixels and cached vectors, not re-extraction."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';study=base/'frozen-grade-correspondence-v1';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(study/'results.json');before=digest(study/'results.json');bridge=base/'frozen-grade-bridge-v1/results.json'
    if not r['finished'] or r['app_promoted'] or r['actual_independent_eligibility'] or digest(bridge)!=r['bridge_results_sha256'] or digest(study/'features.npz')!=r['features_sha256'] or digest(data/'cohort.json')!=r['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256'] or digest(base/'multiref-control-v1/epoch-6.pth')!=r['checkpoint_sha256']:raise ValueError('Changed diagnostic snapshot')
    for name,sha in r['source_sha256'].items():
        if digest(study/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed diagnostic source')
    provenance=r['grade_provenance']
    if digest(ROOT/'.tools/CanonCGT/pretrained/E2E.pth')!=provenance['checkpoint_sha256'] or any(digest(ROOT/'.tools/CanonCGT'/n)!=sha for n,sha in provenance['upstream_source_sha256'].items()) or digest(Path(__file__).with_name('canonical_model.py'))!=provenance['loader_sha256']:raise ValueError('Changed grade provenance')
    rows=load(data/'cohort.json')['groups']['train'];lookup={v['original_path']:v for v in load(data/'srgb-v1/manifest.json')['files']};groups={};count=0
    for i,row in enumerate(rows):
        groups.setdefault(row['group_id'],[]).append(i)
        for role in ['reference','gt']:
            item=lookup[row[role]];path=data/item['prepared_path']
            if digest(path)!=item['sha256']:raise ValueError('Changed parent')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            if r['pixel_records'][count]!=dict(path=row[role],parent_sha256=item['sha256'],pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()):raise ValueError('Input pixel order differs')
            count+=1
    if count!=2048 or len(groups)!=256:raise ValueError('Incomplete pixel audit')
    out=base/'frozen-grade-audit-v1';out.mkdir(exist_ok=False);(out/'audit_frozen_grade.py').write_bytes(Path(__file__).read_bytes());report=dict(scope='Cache-distance and input-pixel audit, no neural feature re-extraction',results_sha256=before,source_sha256=digest(Path(__file__)),input_pixels_verified=count,families={},finished=False,app_promoted=False)
    with np.load(study/'features.npz') as cache:arrays={k:cache[k] for k in ['H128','G64']}
    permutations=list(itertools.permutations(range(4)))
    for name,values in arrays.items():
        if values.shape!=(2048,r['families'][name]['dimensions']) or not np.isfinite(values).all():raise ValueError('Invalid feature cache')
        values=values.astype(float);norm=np.linalg.norm(values,axis=1,keepdims=True);x=values/np.maximum(norm,1e-12);stats=[]
        expected_norm=dict(minimum=float(norm.min()),median=float(np.median(norm)),maximum=float(norm.max()),zero_vectors=int((norm<=1e-12).sum()))
        if expected_norm!=r['families'][name]['norm_summary']:raise ValueError('Norm metrics differ')
        for saved,(group_id,indices) in zip(r['families'][name]['groups'],groups.items()):
            order=sorted(indices,key=lambda i:rows[i]['style_index']);cost=np.clip(1-x[[i*2 for i in order]]@x[[i*2+1 for i in order]].T,0,2)
            if saved['group_id']!=group_id or not np.array_equal(cost,np.asarray(saved['cost'])):raise ValueError('Costs differ')
            ties=np.isclose(cost,cost.min(1)[:,None],rtol=1e-10,atol=1e-12);sums=np.array([sum(cost[i,p[i]] for i in range(4)) for p in permutations]);best=np.flatnonzero(np.isclose(sums,sums.min(),rtol=1e-10,atol=1e-12));diag=float(np.trace(cost)/4);off=float(cost[~np.eye(4,dtype=bool)].mean())
            metric=dict(row_top1=float(np.mean(ties.diagonal()/ties.sum(1))),assignment_fraction_correct=float(np.mean([sum(i==p for i,p in enumerate(permutations[j]))/4 for j in best])),identity_optimum_fraction=float((0 in best)/len(best)),optimal_assignments=len(best),diagonal_mean=diag,offdiagonal_mean=off,margin=off-diag)
            if any(saved[k]!=v for k,v in metric.items()):raise ValueError('Tie-aware scores differ')
            stats.append(metric)
        summary={k:float(np.mean([v[k] for v in stats])) for k in r['families'][name]['summary']}
        if summary!=r['families'][name]['summary']:raise ValueError('Summary differs')
        report['families'][name]=dict(groups=256,maximum_cost_difference=0,summary=summary)
    h,g=[report['families'][n]['summary'] for n in ['H128','G64']];checks=dict(grade_top1_at_least_half=g['row_top1']>=.5,grade_top1_gain_at_least_five_points=g['row_top1']>=h['row_top1']+.05,grade_assignment_no_worse=g['assignment_fraction_correct']>=h['assignment_fraction_correct'])
    if checks!=r['research_value_checks'] or all(checks.values())!=r['may_design_training_comparison'] or digest(study/'results.json')!=before or digest(study/'features.npz')!=r['features_sha256']:raise ValueError('Research-value decision changed')
    report.update(finished=True,passed=True,research_value_checks=checks,may_design_G64_training_comparison=all(checks.values()),actual_independent_eligibility=False);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
