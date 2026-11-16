"""Audit saved field-supervision comparison and original-control reproducibility."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from independent_coco import ROOT,digest


def main():
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=b/'semantic-baseline-context-v1';r=json.loads((out/'results.json').read_text());assert r['finished']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=b/'cohort-multiref-v1';cohort=json.loads((data/'cohort.json').read_text());lookup={x['original_path']:x for x in json.loads((data/'srgb-v1/manifest.json').read_text())['files']};truth={}
    for row in cohort['groups']['validation']:
        e=lookup[row['gt']];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:truth[row['index']]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).astype(float)
    a=torch.load(out/'source_control/epoch-0.pth',weights_only=True,map_location='cpu')
    z=torch.load(out/'baseline_context/epoch-0.pth',weights_only=True,map_location='cpu')
    assert a.keys()==z.keys() and all(torch.equal(a[k],z[k]) for k in a)
    selected={};checked=0;repeated=0
    for arm in ['source_control','baseline_context']:
        for epoch in range(9):
            folder=out/arm/f'epoch-{epoch}';scores=json.loads((folder/'scores.json').read_text());errors=[];pred={}
            for row in scores['cases']:
                p=folder/f"{row['index']}.png";assert digest(p)==row['sha256']
                with Image.open(p) as im:rgb=np.array(im.convert('RGB')).astype(float)
                pred[row['index']]=rgb;err=float(np.abs(rgb-truth[row['index']]).mean());assert abs(err-row['mae'])<1e-10;errors.append(err);checked+=1
            assert abs(np.mean(errors)-scores['mean'])<1e-10
            groups={}
            for row in cohort['groups']['validation']:groups.setdefault(row['group_id'],{})[row['style_index']]=row['index']
            for pair in scores['paired']:
                a,z=[groups[pair['group_id']][k] for k in [0,1]]
                delta=float(np.abs((pred[z]-pred[a])-(truth[z]-truth[a])).mean());assert abs(delta-pair['delta_mae'])<1e-10
            if epoch==r['arms'][arm]['best_epoch']:selected[arm]=scores
    base=np.array([x['mae'] for x in r['arms']['source_control']['baseline']['cases']]);candidate=np.array([x['mae'] for x in selected['baseline_context']['cases']]);control=np.array([x['mae'] for x in selected['source_control']['cases']])
    evidence=dict(scope='Explicit baseline context versus source-color control, no product integration',results_sha256=digest(out/'results.json'),audit_script_sha256=digest(Path(__file__)),checked_predictions=checked,
        candidate_epoch=r['arms']['baseline_context']['best_epoch'],control_epoch=r['arms']['source_control']['best_epoch'],baseline_mean=float(base.mean()),candidate_mean=float(candidate.mean()),control_mean=float(control.mean()),improvement_percent=float((1-candidate.mean()/base.mean())*100),versus_control_percent=float((1-candidate.mean()/control.mean())*100),improved_over_point_one=int((base-candidate>.1).sum()),severe=int(((candidate-base>5)&(candidate-base>.25*base)).sum()),
        mandatory_baseline_mean_passed=bool(candidate.mean()<=.95*base.mean()),mandatory_control_mean_passed=bool(candidate.mean()<=.97*control.mean()),product_integrated=False)
    (ROOT/'docs/research/2026-11-17-semantic-baseline-context-evidence.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
