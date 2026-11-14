"""Recompute all oracle curves with a separate float64 rational blend formula."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';study=base/'blend-capacity-oracle-v1';data=base/'cohort-multiref-v1';control=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(study/'results.json');before=digest(study/'results.json')
    if not r['finished'] or r['app_promoted'] or r['actual_independent_eligibility']:raise ValueError('Incomplete or incorrectly eligible oracle')
    for name,sha in r['source_sha256'].items():
        if digest(study/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed oracle implementation')
    for item in r['assets']:
        if digest(base/item['path'])!=item['sha256']:raise ValueError('Changed oracle asset')
    if digest(data/'cohort.json')!=r['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Cohort metadata changed')
    s=r['endpoint_snapshot'];snapshots={'control_results_sha256':control/'results.json','selected_checkpoint_sha256':control/'epoch-7.pth','current_checkpoint_sha256':base/'multiref-control-v1/epoch-6.pth','baseline_scores_sha256':control/'baseline-current/scores.json','selected_scores_sha256':control/'epoch-7/scores.json'}
    if any(digest(p)!=s[k] for k,p in snapshots.items()):raise ValueError('Changed endpoint snapshot')
    out=base/'blend-capacity-audit-v1';out.mkdir(exist_ok=False)
    for name in ['audit_blend_capacity.py','audit_consistent_reference.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    cohort=load(data/'cohort.json');rows=cohort['groups']['validation'];lookup={v['original_path']:v for v in load(data/'srgb-v1/manifest.json')['files']};a={};b={};truth={};per={};curves=[]
    def png(path):
        with Image.open(path) as im:return np.array(im.convert('RGB'))
    def blend(x,y,k):return np.rint(((100-k)*x.astype(np.float64)+k*y.astype(np.float64))/100).astype(np.uint8)
    for row,saved in zip(rows,r['cases']):
        i=row['index']
        if saved['index']!=i:raise ValueError('Oracle case order differs')
        a[i]=png(control/'baseline-current'/f'{i}.png');b[i]=png(control/'epoch-7'/f'{i}.png')
        with Image.open(data/lookup[row['gt']]['prepared_path']) as im:truth[i]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        curve=[float(np.abs(blend(a[i],b[i],k).astype(float)-truth[i]).mean()) for k in range(101)];k=int(np.argmin(curve));curves.append(curve);per[i]=blend(a[i],b[i],k)
        if curve!=saved['rgb_curve'] or k!=saved['share'] or curve[k]!=saved['mae'] or digest(study/'per-case'/f'{i}.png')!=saved['sha256'] or not np.array_equal(per[i],png(study/'per-case'/f'{i}.png')):raise ValueError('Oracle curve/choice/PNG differs')
    mean=np.mean(curves,axis=0);global_k=int(np.argmin(mean));global_predictions={i:blend(a[i],b[i],global_k) for i in a}
    if mean.tolist()!=r['global_rgb_curve'] or global_k!=r['global_share']:raise ValueError('Global oracle choice differs')
    report=dict(results_sha256=before,source_sha256={n:digest(out/n) for n in ['audit_blend_capacity.py','audit_consistent_reference.py']},curves_recomputed=64,shares_per_curve=101,curves_maximum_difference=0,global_share=global_k,procedures={},finished=False,app_promoted=False,actual_independent_eligibility=False)
    for name,predictions in [('current13',a),('C10',b),('per-case',per),('single-global',global_predictions)]:
        scores=score_outputs(rows,predictions,truth);directory=control/('baseline-current' if name=='current13' else 'epoch-7') if name in ['current13','C10'] else study/name
        for v in scores['cases']:
            path=directory/f'{v["index"]}.png';v['sha256']=digest(path)
            if not np.array_equal(predictions[v['index']],png(path)):raise ValueError('Prediction PNG differs')
        expected=r['endpoints'][name] if name in ['current13','C10'] else r['procedures'][name]['scores']
        if scores!=expected:raise ValueError('Full endpoint/oracle metrics differ')
        if name in ['per-case','single-global']:
            gate=response_development_gate(r['endpoints']['current13'],scores,r['endpoints']['C10'])
            if gate!=r['procedures'][name]['descriptive_gate_only'] or r['procedures'][name]['actual_independent_eligibility']:raise ValueError('Descriptive gate differs or incorrectly permits release')
            report['procedures'][name]=dict(mean=scores['mean'],delta_mean=scores['delta_mean'],descriptive_gate_only=gate)
    gate=report['procedures']['per-case']['descriptive_gate_only'];necessary=bool(gate['relative_improvement_vs_control']>=.03 and gate['improved_vs_control']>=36)
    if necessary!=r['necessary_C10_rgb_checks'] or any(min(curve)>min(curve[0],curve[-1]) for curve in curves):raise ValueError('Necessary RGB bound differs')
    if digest(study/'results.json')!=before or any(digest(p)!=s[k] for k,p in snapshots.items()) or any(digest(Path(__file__).with_name(name))!=sha for name,sha in r['source_sha256'].items()):raise ValueError('Frozen evidence changed during audit')
    report.update(finished=True,passed=True,necessary_C10_rgb_checks=necessary);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
