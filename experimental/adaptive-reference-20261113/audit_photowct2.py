"""Score persisted predictions using the original image preparation environment."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'photowct2-screen-v3'
    report=json.loads((out/'results.json').read_text());assert report['finished'] and len(report['rows'])==64
    assert report['first_author_loop_float_exact'] and report['first_rgb_repeat_exact']
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    for n,h in report['author_sha256'].items():assert digest(ROOT/'.tools/PhotoWCT2-audit'/n)==h
    registry_path=base/'regional-correction-inputs-v1/results.json';assert digest(registry_path)==report['registry_sha256']
    registry=json.loads(registry_path.read_text());prepared=base/'photowct2-inputs-v1'
    assert digest(prepared/'results.json')==report['prepared_report_sha256']
    prep=json.loads((prepared/'results.json').read_text());assert len(prep['inputs'])==96
    assert digest(Path(__file__).with_name('prepare_photowct2_inputs.py'))==prep['source_sha256']
    for e in prep['inputs'].values():
        p=prepared/e['path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:a=np.array(im)
        assert a.shape==(256,256,3) and hashlib.sha256(a.tobytes()).hexdigest()==e['pixel_sha256']
    old=base/'gradestudio-screen-v1';assert digest(old/'results.json')==report['baseline_report_sha256']
    old_rows={r['index']:r for r in json.loads((old/'results.json').read_text())['rows']}
    data=base/'cohort-multiref-v1'
    assert digest(data/'cohort.json')==registry['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==registry['prepared_manifest_sha256']
    truths={r['index']:r for r in json.loads((data/'cohort.json').read_text())['groups']['validation']}
    assert {r['index'] for r in report['rows']}==set(truths)
    lookup={e['original_path']:e for e in json.loads((data/'srgb-v1/manifest.json').read_text())['files']}
    cases=[]
    for r in report['rows']:
        e=lookup[truths[r['index']]['gt']];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:gt=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).astype(float)
        errors={}
        for name,path,sha in [('candidate',out/r['result_path'],r['result_sha256']),('baseline',old/r['baseline_path'],r['baseline_sha256'])]:
            assert digest(path)==sha
            with Image.open(path) as im:a=np.array(im)
            assert a.shape==(256,256,3) and a.dtype==np.uint8
            errors[name]=float(np.abs(a.astype(float)-gt).mean())
        assert errors['baseline']==old_rows[r['index']]['outputs']['baseline']['error']
        assert digest(out/str(r['index'])/'raw.png')==r['raw_sha256']
        cases.append(dict(index=r['index'],**errors,result_sha256=r['result_sha256'],raw_sha256=r['raw_sha256']))
    b=np.array([r['baseline'] for r in cases]);c=np.array([r['candidate'] for r in cases])
    summary=dict(baseline_mean=float(b.mean()),candidate_mean=float(c.mean()),relative_improvement=float(1-c.mean()/b.mean()),
        improved=int((b-c>.1).sum()),severe=int(((c-b>5)&(c>b*1.25)).sum()))
    summary['first_gate_passed']=bool(summary['relative_improvement']>=.05 and summary['improved']>=40 and summary['severe']<=2)
    evidence=dict(scope=report['scope'],summary=summary,cases=cases,verified_prediction_pngs=128,verified_baseline_pngs=64,
        fixed_input_pixels_verified=96,baseline_errors_exact_to_prior=True,first_author_loop_float_exact=True,first_rgb_repeat_exact=True,
        author_sha256=report['author_sha256'],source_sha256=report['source_sha256'],auditor_sha256=digest(Path(__file__)),
        results_sha256=digest(out/'results.json'),prepared_report_sha256=report['prepared_report_sha256'],
        median_cpu_seconds=float(np.median([r['seconds'] for r in report['rows']])),product_integrated=False)
    (ROOT/'docs/research/2026-11-17-photowct2-screen-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
