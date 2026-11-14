"""Postfreeze arithmetic and source-movement audit; does not select coefficients."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from self_bias import cancel


def load(path):
    with Image.open(path) as image:return np.array(image.convert('RGB'))


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/self-bias-20261115-v1'
    report=json.loads((out/'results.json').read_text(encoding='utf-8'))
    natural=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1'
    nr=json.loads((natural/'results.json').read_text(encoding='utf-8'));nl={r['index']:r for r in nr['cases']}
    coco=ROOT/'outputs/adaptive-reference-20261113/independent-coco'
    jobs=json.loads((coco/'development/jobs.json').read_text(encoding='utf-8'))['jobs']
    previous=ROOT/'outputs/adaptive-reference-20261113/region-histogram-20261115-v1'
    pl={r['index']:r for r in json.loads((previous/'results.json').read_text(encoding='utf-8'))['cases']}
    if sum(r['group']=='natural' for r in report['cases'])!=32 or sum(r['group']=='coco' for r in report['cases'])!=30:raise ValueError('Wrong actual32/30 cohort')
    cases=[]
    for row in report['cases']:
        if row['group']=='natural':
            original=nl[row['index']];source=natural/original['outputs']['input']['path'];ref=natural/'srgb'/(original['triplet']['reference']+'.png')
        else:
            job=jobs[row['index']];source=coco/f"{job['target']}-input.png";ref=coco/job['reference_file']
            if digest(ref)!=pl[row['index']]['inputs_sha256']['reference']:raise ValueError('COCO reference differs from previous frozen study')
        if digest(source)!=row['inputs_sha256']['target'] or digest(ref)!=row['inputs_sha256']['reference']:raise ValueError('Input hash changed')
        x=load(source)
        for name,entry in row['files'].items():
            if digest(out/entry['path'])!=entry['sha256']:raise ValueError('Output hash changed')
        with np.load(out/row['files']['raw']['path'],allow_pickle=False) as raw:
            cross,self_raw=raw['cross'],raw['self']
            actual=dict(baseline=np.rint(np.clip(cross,0,1)*255).astype(np.uint8),candidate=cancel(x,cross,self_raw),self=np.rint(np.clip(self_raw,0,1)*255).astype(np.uint8))
            if not np.array_equal(cancel(x,self_raw,self_raw),x):raise ValueError('Algebraic identity failed')
        for name,value in actual.items():
            if not np.array_equal(value,load(out/row['files'][name]['path'])):raise ValueError('Declared raw formula does not reproduce image')
        if row['group']=='coco':
            baseline=previous/pl[row['index']]['outputs']['baseline']['file']
            if digest(baseline)!=pl[row['index']]['outputs']['baseline']['sha256'] or not np.array_equal(actual['baseline'],load(baseline)):raise ValueError('Previous actual baseline changed')
        cases.append(dict(group=row['group'],index=row['index'],source_to_baseline_mae=float(np.abs(x.astype(float)-actual['baseline']).mean()),source_to_candidate_mae=float(np.abs(x.astype(float)-actual['candidate']).mean()),all_three_raw_formulas_exact=True,self_residual_identity_exact=True,previous_coco_baseline_exact=row['group']=='coco'))
    if any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Frozen source changed')
    audit=dict(passed=True,scope='Postfreeze arithmetic/input/cohort/source-movement audit, not new predictions or acceptance',results_sha256=digest(out/'results.json'),verifier_sha256=digest(Path(__file__)),cases=cases,summary={group:{key:float(np.mean([r[key] for r in cases if r['group']==group])) for key in ['source_to_baseline_mae','source_to_candidate_mae']} for group in ['natural','coco']},timing_caveat='Producer cross_seconds includes a second byte render for parity; not standalone baseline latency',cohort_guard_caveat='Producer guard checks total62; this audit verifies actual32natural+30coco')
    (out/'postfreeze-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=len(cases),summary=audit['summary']),indent=2))


if __name__=='__main__':main()
