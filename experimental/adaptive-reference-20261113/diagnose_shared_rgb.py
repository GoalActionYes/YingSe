"""Exact all64 lower bounds, original preparation, no new image prediction."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from shared_rgb import analyze


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';prior=base/'reference-original-extension-v2/results.json';r=json.loads(prior.read_text());assert r['finished']
    data=base/'cohort-training-coverage-v2';manifest=data/'srgb-v1/manifest.json';assert digest(manifest)==r['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in json.loads(manifest.read_text())['files']}
    out=base/'shared-rgb-v1';out.mkdir(exist_ok=False)
    report=dict(scope='Consumed train64; GT-assisted exact point-function lower bound, not normal inference',prior_sha256=digest(prior),source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['shared_rgb.py','diagnose_shared_rgb.py','shared-rgb-protocol.md','test_shared_rgb.py']},cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def read(p):
        e=lookup[p];path=data/e['prepared_path'];assert digest(path)==e['sha256']
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for i,(row,case) in enumerate(zip(r['selected'],r['cases'])):
        assert row['index']==case['index']
        stats=analyze(*[read(row[k]) for k in ['reference_original','reference','content','gt']])
        severe=case['transfer_mae']-case['baseline_mae']>5 and case['transfer_mae']>1.25*case['baseline_mae']
        report['cases'].append(dict(index=row['index'],previous_severe=severe,**stats))
        if (i+1)%16==0:print('checked',i+1,flush=True)
    keys=['joint_minimum_mae','within_weighted_minimum_mae','cross_image_excess_minimum_mae','shared_rgb_keys','reference_shared_pixel_fraction','target_shared_pixel_fraction']
    report['summary']={k:dict(mean=float(np.mean([c[k] for c in report['cases']])),median=float(np.median([c[k] for c in report['cases']])),maximum=float(max(c[k] for c in report['cases']))) for k in keys}
    report['severe_cases']=[c for c in report['cases'] if c['previous_severe']]
    assert len(report['cases'])==64 and len(report['severe_cases'])==3
    report.update(finished=True,training=False,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    (ROOT/'docs/research/2026-11-17-shared-rgb-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=report['summary'],severe=report['severe_cases'])))


if __name__=='__main__':main()
