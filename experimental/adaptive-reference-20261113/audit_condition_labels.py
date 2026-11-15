"""CPU audit all train-only condition labels; never launches fitting or prediction."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'condition-labels-train-v1';data=base/'cohort-multiref-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'));p=out/'results.json';report=load(p);cohort=load(data/'cohort.json');lookup={r['original_path']:r for r in load(data/'srgb-v1/manifest.json')['files']}
    if not report['finished'] or report['split']!='train' or report['development_labels_used'] or len(report['cases'])!=1024:raise ValueError('Incomplete or mixed split')
    if digest(data/'cohort.json')!=report['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=report['prepared_manifest_sha256']:raise ValueError('Source data changed')
    if {r['index'] for r in report['cases']}&{r['index'] for r in cohort['groups']['validation']}:raise ValueError('Validation indices leaked')
    for n,sha in report['source_sha256'].items():
        if digest(Path(__file__).with_name(n))!=sha or digest(out/'frozen-source'/n)!=sha:raise ValueError('Frozen source changed')
    a=[];b=[];boundary=0;steps={}
    for case,row in zip(report['cases'],cohort['groups']['train']):
        if (case['index'],case['group_id'],case['style_index'])!=(row['index'],row['group_id'],row['style_index']):raise ValueError('Training order mismatch')
        for role in ['content','reference','gt']:
            entry=lookup[row[role]];stored=case['inputs'][role]
            if stored['path']!=row[role] or digest(data/entry['prepared_path'])!=entry['sha256'] or stored['parent_sha256']!=entry['sha256']:raise ValueError('Input hash/mapping mismatch')
        with Image.open(data/lookup[row['gt']]['prepared_path']) as im:truth=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        for name,key,collection in [('baseline','baseline_mae',a),('gt_fit','fit_mae',b)]:
            entry=case['files'][name];path=out/entry['path']
            if digest(path)!=entry['sha256']:raise ValueError('Output changed')
            pixels=np.asarray(Image.open(path).convert('RGB'));mae=float(np.abs(pixels.astype(float)-truth.astype(float)).mean())
            if pixels.shape!=(256,256,3) or mae!=case[key]:raise ValueError('Saved score mismatch')
            collection.append(mae)
        entry=case['files']['conditions'];path=out/entry['path']
        if digest(path)!=entry['sha256']:raise ValueError('Parameters changed')
        snapshots=case['snapshots'];values=[r['mae'] for r in snapshots];selected=int(np.argmin(values))
        if len(snapshots)!=9 or [r['step'] for r in snapshots]!=list(range(0,129,16)) or snapshots[selected]['step']!=case['selected_step'] or values[0]!=case['baseline_mae'] or values[selected]!=case['fit_mae']:raise ValueError('Snapshot selection mismatch')
        with np.load(path) as z:
            d=z['snapshot_deltas']
            if d.shape!=(9,128) or d.dtype!=np.float32 or not np.isfinite(d).all() or (np.abs(d)>3).any() or d[0].any() or not np.array_equal(z['snapshot_steps'],np.arange(0,129,16)) or not np.array_equal(z['selected_delta'],d[selected]):raise ValueError('Invalid target parameters')
            boundary+=int(np.abs(d[selected]).max()>=3)
        steps[str(case['selected_step'])]=steps.get(str(case['selected_step']),0)+1
    a=np.asarray(a);b=np.asarray(b);summary=dict(count=1024,baseline_mean=float(a.mean()),fit_mean=float(b.mean()),improved=int((b<a-.1).sum()),regressed=int((b>a+.1).sum()))
    if summary!=report['summary'] or report['backbone_state_before']!=report['backbone_state_after']:raise ValueError('Aggregate/state mismatch')
    result=dict(scope='All1024 train labels CPU artifact audit, no GPU replay or normal inference acceptance',producer_results_sha256=digest(p),auditor_sha256=digest(Path(__file__)),input_role_records_checked=3072,pngs_scored=2048,condition_snapshots_checked=9216,summary=summary,boundary_selected=boundary,selected_steps=steps,development_labels_used=False,finished=True)
    dst=out/'artifact-audit.json'
    if dst.exists():raise ValueError('Refusing overwrite')
    dst.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
