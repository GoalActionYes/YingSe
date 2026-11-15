"""Posthoc GT-selected candidate-space upper bound; never a deployable selector."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from probe_petal import jobs,working,load


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/petal-retouch-20261115-v1'
    source=base/'results.json';parent=json.loads(source.read_text(encoding='utf-8'))
    if not parent['finished'] or len(parent['cases'])!=77:raise ValueError('Main experiment not complete')
    out=base/'oracle-posthoc';out.mkdir(exist_ok=False)
    names=['analyze_petal_retouch_oracle.py','petal-retouch-oracle-protocol.md']
    for name in names:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Posthoc consumed77 candidate-space upper bound; GT used to select, not deployable, not a product score',main_results_sha256=digest(source),source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},truth_used_for_selection=True,all_four_options_allowed_even_if_main_safety_guard_rejects=True,product_integrated=False,cases=[])
    for row,job in zip(parent['cases'],jobs()):
        if (row['group'],row['index'])!=(job['group'],job['index']):raise ValueError('Ordering changed')
        truth=working(job['truth'],job['group']);target=working(job['target'],job['group'])
        if job['group']=='natural':truth=np.array(Image.fromarray(truth).resize((target.shape[1],target.shape[0]),Image.Resampling.BILINEAR))
        if digest(job['truth'])!=row['truth_sha256']:raise ValueError('Truth changed')
        values=[]
        for name in ['baseline','option-1','option-2','option-3']:
            entry=row['files'][name];p=base/entry['path']
            if digest(p)!=entry['sha256']:raise ValueError('Option changed')
            values.append(float(np.abs(load(p).astype(np.float64)-truth.astype(np.float64)).mean()))
        if values[0]!=row['scores']['baseline']['mae'] or values[row['choice']]!=row['scores']['candidate']['mae']:raise ValueError('Main score mismatch')
        report['cases'].append(dict(group=row['group'],index=row['index'],people=row['people'],profile=row['profile'],actual_choice=row['choice'],oracle_choice=int(np.argmin(values)),option_mae=values,actual_mae=values[row['choice']],oracle_mae=min(values),baseline_mae=values[0]))
    def aggregate(rows,epsilon):
        a=np.array([r['baseline_mae'] for r in rows]);b=np.array([r['oracle_mae'] for r in rows]);c=np.array([r['actual_mae'] for r in rows]);available=b<a-epsilon
        return dict(count=len(rows),baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),actual_mean=float(c.mean()),improved=int(available.sum()),regressed=int((b>a+epsilon).sum()),severe=int(((b>a+5)&(b>a*1.25)).sum()),no_available_improvement=int((~available).sum()),missed_any_available_improvement=int((available&(c>=a-epsilon)).sum()),selected_regression_despite_available_improvement=int((available&(c>a+epsilon)).sum()),selection_regret_mean=float((c-b).mean()))
    by={g:[r for r in report['cases'] if r['group']==g] for g in ['pst','natural','coco']};regional=[r for r in by['coco'] if r['people']!='none' and r['profile'].startswith('region_')];controls=[r for r in by['coco'] if r not in regional]
    s={g:aggregate(rows,.5 if g=='coco' else .1) for g,rows in by.items()};s.update(regional=aggregate(regional,.5),controls=aggregate(controls,.5))
    s['passed_gate']=all(s[g]['candidate_mean']<=.95*s[g]['baseline_mean'] and s[g]['improved']>=n and s[g]['severe']<=severe for g,n,severe in [('natural',22,2),('pst',10,1)]) and s['regional']['candidate_mean']<=.9*s['regional']['baseline_mean'] and s['regional']['improved']>=6 and s['controls']['regressed']<=2 and s['coco']['severe']==0
    report['summary']=s
    if digest(source)!=report['main_results_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Analysis evidence changed')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(s,indent=2),flush=True)


if __name__=='__main__':main()
