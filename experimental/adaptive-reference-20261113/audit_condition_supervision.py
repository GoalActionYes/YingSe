"""CPU artifact audit. Does not resume training or overwrite producer results."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'condition-supervision-training-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    producer=load(out/'results.json');cohort=load(base/'cohort-multiref-v1/cohort.json');prep=load(base/'cohort-multiref-v1/srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']}
    post=load(out/'postrun-source.json') if (out/'postrun-source.json').exists() else None
    for name,sha in producer['source_sha256'].items():
        current=digest(Path(__file__).with_name(name))
        allowed=post is not None and name==post['changed_file'] and sha==post['run_source_sha256'] and current==post['current_source_sha256']
        if digest(out/'frozen-source'/name)!=sha or (current!=sha and not allowed):raise ValueError('Training source changed without recorded postrun hardening')
    truth={}
    for row in cohort['groups']['validation']:
        entry=lookup[row['gt']];p=base/'cohort-multiref-v1'/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('GT changed')
        with Image.open(p) as im:truth[row['index']]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    all_scores={};artifacts={};initial_states=[]
    for arm in ['pixel-only','condition-supervised']:
        records=producer['arms'][arm]['epochs']
        if [r['updates'] for r in records]!=[256*i for i in range(1,9)]:raise ValueError('Incomplete training budget')
        epochs=[]
        for epoch in range(9):
            cp=out/arm/f'epoch-{epoch}.pth';state=torch.load(cp,map_location='cpu',weights_only=True)
            if set(state)!={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias'} or any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Invalid head checkpoint')
            if epoch==0:initial_states.append(state)
            elif digest(cp)!=records[epoch-1]['head_sha256']:raise ValueError('Head hash mismatch')
            path=out/arm/f'epoch-{epoch}/scores.json';scores=load(path);paired={};cases=[]
            for row,case in zip(cohort['groups']['validation'],scores['cases']):
                p=path.parent/f'{row["index"]}.png'
                if case['index']!=row['index'] or case['style_index']!=row['style_index'] or digest(p)!=case['sha256']:raise ValueError('Prediction linkage mismatch')
                a=np.asarray(Image.open(p).convert('RGB'));gt=truth[row['index']];mae=float(np.abs(a.astype(float)-gt.astype(float)).mean())
                if mae!=case['mae']:raise ValueError('Score mismatch')
                cases.append(case);paired.setdefault(row['group_id'],{})[row['style_index']]=(a.astype(float),gt.astype(float))
            means=float(np.mean([r['mae'] for r in cases]));delta=float(np.mean([np.abs((v[1][0]-v[0][0])-(v[1][1]-v[0][1])).mean() for v in paired.values()]))
            if len(cases)!=64 or len(paired)!=32 or means!=scores['mean'] or delta!=scores['delta_mean']:raise ValueError('Aggregate mismatch')
            if epoch and (means!=records[epoch-1]['validation_mean'] or delta!=records[epoch-1]['delta_mean']):raise ValueError('Epoch record mismatch')
            epochs.append(scores);artifacts[str(path.relative_to(out))]=digest(path)
        best=min(range(9),key=lambda i:epochs[i]['mean'])
        if best!=producer['arms'][arm]['best_epoch']:raise ValueError('Checkpoint selection mismatch')
        all_scores[arm]=dict(best_epoch=best,selected=epochs[best],baseline=epochs[0])
    if any(not torch.equal(initial_states[0][k],initial_states[1][k]) for k in initial_states[0]):raise ValueError('Initial heads differ')
    a=all_scores['condition-supervised']['baseline'];b=all_scores['condition-supervised']['selected'];c=all_scores['pixel-only']['selected']
    if a!=all_scores['pixel-only']['baseline']:raise ValueError('Zero baselines differ')
    def comparison(before,after,ratio,count):
        x=np.array([r['mae'] for r in before['cases']]);y=np.array([r['mae'] for r in after['cases']]);improved=int((y<x-.1).sum());severe=int(((y>x+5)&(y>x*1.25)).sum())
        return dict(before_mean=before['mean'],after_mean=after['mean'],improved=improved,severe=severe,passed=bool(after['mean']<=before['mean']*(1-ratio) and improved>=count and severe<=2))
    gates=dict(vs_current=comparison(a,b,.05,40),vs_matched=comparison(c,b,.03,36))
    x=np.array([r['mae'] for r in a['cases'] if r['style_index']==1]);y=np.array([r['mae'] for r in b['cases'] if r['style_index']==1])
    gates['added_reference_passed']=bool(y.mean()<=x.mean() and (y<x-.1).sum()>=17)
    x=np.array([r['delta_mae'] for r in c['paired']]);y=np.array([r['delta_mae'] for r in b['paired']]);gates['response_vs_matched_passed']=bool(y.mean()<=.95*x.mean() and (y<x-.1).sum()>=20)
    gates['passed']=bool(gates['vs_current']['passed'] and gates['vs_matched']['passed'] and gates['added_reference_passed'] and gates['response_vs_matched_passed'])
    result=dict(scope='Independent CPU saved-artifact audit; no GPU rerender or independent acceptance',producer_report_sha256=digest(out/'results.json'),producer_finished_flag=producer['finished'],source_sha256=digest(Path(__file__)),audited_png_count=1152,audited_head_count=18,score_artifacts_sha256=artifacts,arms=all_scores,gates=gates,finished=True)
    if post is not None:result['postrun_source_hardening']=post
    dst=out/('artifact-audit-posthardening.json' if post else 'artifact-audit.json')
    if dst.exists():raise ValueError('Refusing to overwrite audit')
    dst.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(selected={n:v['best_epoch'] for n,v in all_scores.items()},gates=gates)),flush=True)


if __name__=='__main__':main()
