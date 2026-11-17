"""Independently re-read diagnostic PNGs, GT, selection and coefficients."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'reference-original-extension-v2'
    load=lambda p:json.loads(p.read_text());r=load(out/'results.json');assert r['finished'] and len(r['cases'])==64
    data=base/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json')
    assert digest(data/'cohort.json')==r['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==r['prepared_manifest_sha256']
    lookup={e['original_path']:e for e in prep['files']};rows={e['index']:e for e in cohort['groups']['train']}
    coverage=base/'reference-original-coverage-v1';assert digest(coverage/'results.json')==r['coverage_report_sha256']
    old={e[0] for e in load(coverage/'cohort-multiref-v1.json')['selected']}
    new=load(coverage/'cohort-training-coverage-v2.json')['selected'];selected=[];seen=set()
    for index,original,reference in new:
        row=rows[index]
        if index in old or row['content'] in seen:continue
        selected.append(dict(row,reference_original=original));seen.add(row['content'])
        if len(selected)==64:break
    assert selected==r['selected'] and len(seen)==64
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h and digest(out/'frozen-source'/n)==h
    assert digest(base/'multiref-control-v1/epoch-6.pth')==r['checkpoint_sha256']
    assert r['bridge']['three_outputs_pixel_exact']
    def read(p,sha):
        assert digest(p)==sha
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def image(relative):
        e=lookup[relative];return read(data/e['prepared_path'],e['sha256'])
    cases=[]
    for row,case in zip(selected,r['cases']):
        assert case['index']==row['index']
        gt=image(row['gt']);reference=image(row['reference']);scores={}
        for label in ['baseline','transfer','reference-fit']:
            e=case['outputs'][label];p=out/e['path'];a=read(p,e['sha256'])
            expected=reference if label=='reference-fit' else gt
            scores[label]=float(np.abs(a.astype(float)-expected).mean())
        assert scores['baseline']==case['baseline_mae'] and scores['transfer']==case['transfer_mae'] and scores['reference-fit']==case['reference_crossfit_mae']
        p=out/str(row['index'])/'coefficients.npy';assert digest(p)==case['coefficients_sha256']
        coefficients=np.load(p,allow_pickle=False);assert coefficients.shape==(32,3) and np.isfinite(coefficients).all()
        cases.append(dict(index=row['index'],**scores,output_sha256={k:v['sha256'] for k,v in case['outputs'].items()},coefficients_sha256=case['coefficients_sha256']))
    b=np.array([v['baseline'] for v in cases]);c=np.array([v['transfer'] for v in cases])
    summary=dict(baseline_mean=float(b.mean()),transfer_mean=float(c.mean()),relative_improvement=float(1-c.mean()/b.mean()),improved=int((b-c>.1).sum()),severe=int(((c-b>5)&(c>b*1.25)).sum()),reference_crossfit_mean=float(np.mean([v['reference-fit'] for v in cases])))
    summary['supervision_probe_passed']=bool(summary['relative_improvement']>=.05 and summary['improved']>=40 and summary['severe']<=2)
    assert summary==r['summary']
    evidence=dict(scope=r['scope'],summary=summary,cases=cases,results_sha256=digest(out/'results.json'),
        source_sha256=r['source_sha256'],auditor_sha256=digest(Path(__file__)),checkpoint_sha256=r['checkpoint_sha256'],
        bridge=r['bridge'],predictions_verified=192,coefficients_verified=64,selection_exact=True,training=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-reference-original-extension-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
