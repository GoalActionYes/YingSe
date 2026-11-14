"""All32 consumed reference pairs: exact score bridge and descriptive geometry."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from response_geometry import response_geometry
from audit_consistent_reference import score_outputs


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');audit=load(base/'training-coverage-audit-v2/results.json')
    if not audit['passed'] or not prep['passed'] or digest(data/'srgb-v1/manifest.json')!=audit['prepared_manifest_sha256']:raise ValueError('Completed audit/data changed')
    lookup={r['original_path']:r for r in prep['files']};rows=cohort['groups']['validation'];groups=defaultdict(dict)
    for r in rows:
        if r['style_index'] in groups[r['group_id']]:raise ValueError('Repeated pair identity')
        groups[r['group_id']][r['style_index']]=r
    if len(groups)!=32 or any(set(v)!={0,1} for v in groups.values()):raise ValueError('Incomplete32pairs')
    prior=base/'appearance-pairwise-replica-v1';directories=dict(current13=prior/'baseline-current',C10=prior/'epoch-7')
    protected={str(p.relative_to(base)):digest(p) for p in [data/'cohort.json',data/'srgb-v1/manifest.json',base/'training-coverage-audit-v2/results.json']}
    for k in ('control','expanded'):
        p=base/f'training-coverage-{k}-v2/results.json';r=load(p)
        if not r['finished'] or digest(p)!=audit['results_sha256'][k]:raise ValueError('Selected trial changed')
        directories[k]=p.parent/f'epoch-{r["best_epoch"]}';protected[str(p.relative_to(base))]=digest(p)
    truth={}
    for r in rows:
        item=lookup[r['gt']];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('GT changed')
        with Image.open(p) as im:truth[r['index']]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    tables={};pixels={}
    for k,directory in directories.items():
        scorepath=directory/'scores.json';saved=load(scorepath);protected[str(scorepath.relative_to(base))]=digest(scorepath);pred={}
        for r in saved['cases']:
            p=directory/f'{r["index"]}.png'
            if digest(p)!=r['sha256']:raise ValueError('Prediction changed')
            with Image.open(p) as im:pred[r['index']]=np.array(im.convert('RGB'))
        actual=score_outputs(rows,pred,truth)
        for r in actual['cases']:r['sha256']=digest(directory/f'{r["index"]}.png')
        if actual!=saved:raise ValueError('Full RGB/response/tone bridge differs')
        tables[k]={r['group_id']:r for r in saved['paired']};pixels[k]=pred
    cases=[]
    for name,group in groups.items():
        a,b=[group[i]['index'] for i in (0,1)];modes={}
        for k in directories:
            geometry=response_geometry(pixels[k][a],pixels[k][b],truth[a],truth[b]);paired=tables[k][name]
            if geometry['gain']!=paired['response_gain']:raise ValueError('Projection gain differs from old metric')
            modes[k]=dict(geometry=geometry,paired=paired)
        change=tables['expanded'][name]['delta_mae']-tables['C10'][name]['delta_mae'];cases.append(dict(group_id=name,indices=[a,b],delta_error_change_vs_C10=change,stratum='regressed' if change>.1 else 'improved' if change<-.1 else 'near_zero',modes=modes))
    summaries={}
    for stratum in ('all','improved','regressed','near_zero'):
        selected=[r for r in cases if stratum=='all' or r['stratum']==stratum];summary=dict(groups=len(selected),modes={})
        for k in directories:
            valid=lambda key:[r['modes'][k]['geometry'][key] for r in selected if r['modes'][k]['geometry'][key] is not None]
            summary['modes'][k]=dict(mean_delta_mae=float(np.mean([r['modes'][k]['paired']['delta_mae'] for r in selected])),median_geometry={key:float(np.median(valid(key))) if valid(key) else None for key in ('gain','amplitude_ratio','cosine','orthogonal_ratio')},defined_geometry_counts={key:len(valid(key)) for key in ('gain','amplitude_ratio','cosine','orthogonal_ratio')})
        comparable=[r for r in selected if all(r['modes'][k]['geometry']['gain'] is not None for k in ('C10','expanded'))]
        summary['comparison']=dict(gain_decreased=sum(r['modes']['expanded']['geometry']['gain']<r['modes']['C10']['geometry']['gain'] for r in comparable),orthogonal_increased=sum(r['modes']['expanded']['geometry']['orthogonal_ratio']>r['modes']['C10']['geometry']['orthogonal_ratio'] for r in comparable),expanded_negative_gain=sum(r['modes']['expanded']['geometry']['gain']<0 for r in comparable))
        summaries[stratum]=summary
    if [summaries[k]['groups'] for k in ('improved','regressed','near_zero')]!=[18,11,3]:raise ValueError('Original strata changed')
    names=['diagnose_coverage_response.py','response_geometry.py','response-geometry-protocol.md','multiref_response.py','tone_response.py','audit_consistent_reference.py'];sources={n:digest(Path(__file__).with_name(n)) for n in names}
    if any(digest(base/n)!=sha for n,sha in protected.items()):raise ValueError('Frozen scores changed')
    out=base/'coverage-response-geometry-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    result=dict(scope='Descriptive all32consumed development pairs; no causal/semantic/independent acceptance or training',source_sha256=sources,protected_inputs_sha256=protected,cases=cases,summaries=summaries,full_score_bridges_verified=4,prediction_png_verified=256,prepared_gt_exposures_verified=64,finished=True,new_training_started=False,app_promoted=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
