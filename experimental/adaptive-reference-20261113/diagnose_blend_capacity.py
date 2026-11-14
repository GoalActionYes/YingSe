"""Two GT-dependent blend procedures; no model training or release candidate."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest
from blend_capacity import blend_codes,rgb_oracle
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';control=base/'appearance-pairwise-replica-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));record=load(control/'results.json')
    checkpoint=control/'epoch-7.pth';current=base/'multiref-control-v1/epoch-6.pth';cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    if not record['finished'] or not record['appearance_enabled'] or record['best_epoch']!=7 or digest(control/'results.json')!='3655c1120a7d854269304810c3be144e281d871a4cd6d1cd62c2236e2b7d093c' or digest(checkpoint)!=record['selected_checkpoint_sha256'] or digest(current)!=record['start_checkpoint_sha256'] or digest(cohort_path)!=record['cohort_sha256'] or digest(prep_path)!=record['prepared_manifest_sha256']:raise ValueError('Changed frozen endpoints/data')
    for name,sha in record['source_sha256'].items():
        if digest(control/'frozen-source'/name)!=sha or digest(Path(__file__).with_name(name))!=sha:raise ValueError('Changed endpoint implementation')
    cohort,prep=load(cohort_path),load(prep_path);rows=cohort['groups']['validation'];lookup={v['original_path']:v for v in prep['files']}
    if len(rows)!=64 or len({v['index'] for v in rows})!=64:raise ValueError('Expected consumed64 development')
    endpoints={'current13':load(control/'baseline-current/scores.json'),'C10':load(control/'epoch-7/scores.json')}
    if endpoints['current13']!=record['baseline']:raise ValueError('Changed baseline scores')
    out=base/'blend-capacity-oracle-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();(out/'per-case').mkdir();(out/'single-global').mkdir()
    names=['diagnose_blend_capacity.py','blend_capacity.py','test_blend_capacity.py','blend-capacity-protocol.md','audit_consistent_reference.py','multiref_response.py','tone_response.py','pairwise_gate.py','training_gates.py','independent_coco.py']
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    snapshot=dict(control_results_sha256=digest(control/'results.json'),selected_checkpoint_sha256=digest(checkpoint),current_checkpoint_sha256=digest(current),baseline_scores_sha256=digest(control/'baseline-current/scores.json'),selected_scores_sha256=digest(control/'epoch-7/scores.json'))
    report=dict(scope='Consumed64 development GT-dependent byte-blend capacity only, never inference or acceptance',endpoint_snapshot=snapshot,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},assets=[],cases=[],finished=False,app_promoted=False,actual_independent_eligibility=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8');a={};b={};truth={};inputs={}
    for number,row in enumerate(rows):
        i=row['index'];values=[]
        for name,label in [('current13','baseline-current'),('C10','epoch-7')]:
            path=control/label/f'{i}.png';entry=next(v for v in endpoints[name]['cases'] if v['index']==i)
            if digest(path)!=entry['sha256']:raise ValueError('Endpoint PNG changed')
            with Image.open(path) as im:values.append(np.array(im.convert('RGB')))
            report['assets'].append(dict(path=str(path.relative_to(base)),sha256=entry['sha256']))
        prepared={}
        for role in ['content','reference','gt']:
            entry=lookup[row[role]];path=data/entry['prepared_path']
            if digest(path)!=entry['sha256']:raise ValueError('Prepared development input changed')
            with Image.open(path) as im:prepared[role]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            report['assets'].append(dict(path=str(path.relative_to(base)),sha256=entry['sha256']))
        a[i],b[i]=values;truth[i]=prepared['gt'];inputs[i]=prepared
        if values[0].shape!=(256,256,3) or values[1].shape!=(256,256,3):raise ValueError('Endpoint dimensions differ')
        curve,k=rgb_oracle(a[i],b[i],truth[i]);path=out/'per-case'/f'{i}.png';Image.fromarray(blend_codes(a[i],b[i],k)).save(path)
        report['cases'].append(dict(index=i,share=k,rgb_curve=curve,mae=curve[k],sha256=digest(path)))
        if (number+1)%16==0:print(json.dumps(dict(cases_scored=number+1)),flush=True)
    for name,arrays in [('current13',a),('C10',b)]:
        actual=score_outputs(rows,arrays,truth)
        for item in actual['cases']:item['sha256']=digest(control/('baseline-current' if name=='current13' else 'epoch-7')/f'{item["index"]}.png')
        if actual!=endpoints[name]:raise ValueError('Endpoint RGB/paired/tone metrics do not reproduce')
    mean_curve=np.mean([v['rgb_curve'] for v in report['cases']],axis=0);global_k=int(np.argmin(mean_curve));per={v['index']:blend_codes(a[v['index']],b[v['index']],v['share']) for v in report['cases']};global_predictions={i:blend_codes(a[i],b[i],global_k) for i in a}
    for i,pixels in global_predictions.items():Image.fromarray(pixels).save(out/'single-global'/f'{i}.png')
    report['procedures']={}
    for name,predictions in [('per-case',per),('single-global',global_predictions)]:
        scores=score_outputs(rows,predictions,truth)
        for item in scores['cases']:item['sha256']=digest(out/name/f'{item["index"]}.png')
        gate=response_development_gate(endpoints['current13'],scores,endpoints['C10']);report['procedures'][name]=dict(scores=scores,descriptive_gate_only=gate,actual_independent_eligibility=False)
    if any(v['mae']>min(v['rgb_curve'][0],v['rgb_curve'][100]) for v in report['cases']):raise ValueError('Oracle failed endpoint lower bound')
    gate=report['procedures']['per-case']['descriptive_gate_only'];report.update(global_share=global_k,global_rgb_curve=mean_curve.tolist(),endpoints=endpoints,necessary_C10_rgb_checks=bool(gate['relative_improvement_vs_control']>=.03 and gate['improved_vs_control']>=36))
    for row in rows:
        i=row['index'];share=next(v['share'] for v in report['cases'] if v['index']==i);canvas=Image.new('RGB',(1024,550),'#222');draw=ImageDraw.Draw(canvas)
        panels=[('Target',inputs[i]['content']),('Reference',inputs[i]['reference']),('GT',truth[i]),('Current13',a[i]),('C10',b[i]),(f'GT global share {global_k}',global_predictions[i]),(f'GT per-image share {share}',per[i])]
        for j,(label,pixels) in enumerate(panels):
            x=j%4*256;y=j//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(pixels),(x,y+19))
        canvas.save(out/f'{i}.jpg',quality=95)
    if digest(control/'results.json')!=snapshot['control_results_sha256'] or digest(checkpoint)!=snapshot['selected_checkpoint_sha256'] or digest(current)!=snapshot['current_checkpoint_sha256'] or digest(control/'baseline-current/scores.json')!=snapshot['baseline_scores_sha256'] or digest(control/'epoch-7/scores.json')!=snapshot['selected_scores_sha256'] or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()) or any(digest(base/v['path'])!=v['sha256'] for v in report['assets']):raise ValueError('Frozen inputs changed during diagnostic')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,global_share=global_k,necessary_C10_rgb_checks=report['necessary_C10_rgb_checks'],procedures={n:dict(mean=v['scores']['mean'],delta_mean=v['scores']['delta_mean'],descriptive_gate_only=v['descriptive_gate_only']) for n,v in report['procedures'].items()})),flush=True)


if __name__=='__main__':main()
