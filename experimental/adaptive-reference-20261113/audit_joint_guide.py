"""Recompute held-out image errors from original GT and verify saved coefficients."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'joint-guide-v1';load=lambda p:json.loads(p.read_text())
    r=load(out/'results.json');assert r['finished'] and len(r['cases'])==64
    prior=base/'reference-original-extension-v2/results.json';assert digest(prior)==r['prior_sha256'];old=load(prior)
    data=base/'cohort-training-coverage-v2';manifest=data/'srgb-v1/manifest.json';assert digest(manifest)==old['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in load(manifest)['files']}
    assert digest(base/'multiref-control-v1/epoch-6.pth')==r['checkpoint_sha256']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==digest(out/'frozen-source'/n)==h
    def truth(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).astype(float)
    for row,c,p in zip(old['selected'],r['cases'],old['cases']):
        assert row['index']==c['index']==p['index']
        assert c['baseline_mae']==p['baseline_mae'] and c['source_only_target_mae']==p['transfer_mae'] and c['source_only_reference_mae']==p['reference_crossfit_mae']
        gt,ref=truth(row['gt']),truth(row['reference'])
        for label,e in c['outputs'].items():
            path=out/str(c['index'])/(label+'.png');assert digest(path)==e['sha256']
            with Image.open(path) as im:a=np.array(im)
            assert a.shape==(256,256,3) and a.dtype==np.uint8
            assert float(np.abs(a.astype(float)-(ref if label=='joint_reference' else gt)).mean())==e['mae']
        for label in ['target_only','joint']:
            path=out/str(c['index'])/(label+'-coefficients.npy');assert digest(path)==c[label+'_coefficients_sha256']
            a=np.load(path,allow_pickle=False);assert a.shape==(2,32,3) and np.isfinite(a).all()
    expected={k:float(np.mean([c['outputs'][k]['mae'] for c in r['cases']])) for k in ['target_only','joint_target','joint_reference']}
    expected.update(source_only_target=float(np.mean([c['source_only_target_mae'] for c in r['cases']])),source_only_reference=float(np.mean([c['source_only_reference_mae'] for c in r['cases']])))
    assert expected==r['summary']
    evidence=dict(scope=r['scope'],summary=expected,cases=r['cases'],results_sha256=digest(out/'results.json'),source_sha256=r['source_sha256'],
        audit_source_sha256=digest(Path(__file__)),png_verified=192,coefficient_archives_verified=128,source_only_pairs_reproduced_in_run=64,
        matrix_native_max_code_difference=max(c['native_max_code_difference'] for c in r['cases']),
        severe_cases=[c for c in r['cases'] if c['previous_severe']],training=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-joint-guide-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=expected,severe=[dict(index=c['index'],target_only=c['outputs']['target_only']['mae'],joint_target=c['outputs']['joint_target']['mae'],joint_reference=c['outputs']['joint_reference']['mae']) for c in evidence['severe_cases']])))


if __name__=='__main__':main()
