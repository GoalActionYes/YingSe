"""Two-fold spatially held-out GT fitting, diagnostic only, all 64 dev cases."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from capacity_audit import block_split,polynomial_features,ridge_fit,rgb_collision_floor,native_features,render_coefficients,post_forward,inverse_post


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='capacity-audit-v1');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Expected directory name')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';current=base/'multiref-control-v1/epoch-6';checkpoint=current.with_suffix('.pth')
    checkpoint_sha='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    if digest(checkpoint)!=checkpoint_sha:raise ValueError('Changed current model')
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';score_path=current/'scores.json'
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));prep=json.loads(prep_path.read_text(encoding='utf-8'));scores=json.loads(score_path.read_text(encoding='utf-8'))
    anchor=json.loads((current.parent/'results.json').read_text(encoding='utf-8'))
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path) or anchor['cohort_sha256']!=digest(cohort_path) or anchor['prepared_manifest_sha256']!=digest(prep_path):raise ValueError('Changed cohort or preparation')
    score_lookup={r['index']:r for r in scores['cases']};lookup={r['original_path']:r for r in prep['files']};groups=defaultdict(dict)
    for row in cohort['groups']['validation']:
        if row['style_index'] in groups[row['group_id']]:raise ValueError('Duplicate style')
        groups[row['group_id']][row['style_index']]=row
    if len(groups)!=32 or any(set(g)!={0,1} for g in groups.values()) or len(score_lookup)!=64:raise ValueError('Incomplete fixed development groups')
    out=base/args.run;out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['run_capacity_audit.py','capacity_audit.py','capacity-audit-protocol.md','train_triplet_pilot.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='GT-assisted fits on consumed development content; two within-image spatial folds, not reference inference or independent acceptance',
        config=dict(size=256,tile=32,ridge=1e-5,folds=2,spatial_nodes=9,guide_layers=8),checkpoint_sha256=checkpoint_sha,cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),
        scores_sha256=digest(score_path),source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,_=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite current weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    def read(relative):
        entry=lookup[relative];path=data/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed prepared image')
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def errors(actual,truth,mask=None):
        delta=np.abs(actual.astype(float)-truth)
        if mask is not None:delta=delta.reshape(-1,3)[mask]
        return dict(mae=float(delta.mean()),p95=float(np.quantile(delta,.95)))
    started=time.perf_counter()
    for number,(group,rows) in enumerate(groups.items()):
        if rows[0]['content']!=rows[1]['content']:raise ValueError('Paired content differs')
        rgb=read(rows[0]['content']);truth=[read(rows[i]['gt']) for i in [0,1]];features=native_features(model,api,rgb);mask=block_split(256,256,32)
        design={'affine':polynomial_features(rgb,1),'quadratic':polynomial_features(rgb,2),'guided_global':features['guided_global'],'guided_spatial':features['guided_spatial']}
        truth_arrays=np.concatenate([x.reshape(-1,3).astype(float)/255 for x in truth],1)
        inverses=[inverse_post(x,features['post_tables']) for x in truth];inverse_arrays=np.concatenate([v[0] for v in inverses],1)
        outputs=[{name:np.empty_like(rgb) for name in design} for _ in [0,1]];details=[{name:[] for name in design} for _ in [0,1]]
        keys=rgb.reshape(-1,3).astype(np.uint32);keys=(keys[:,0]<<16)|(keys[:,1]<<8)|keys[:,2];coverage=[]
        for fold in [0,1]:
            train=mask if fold==0 else ~mask;heldout=~train;coverage.append(float(np.isin(keys[heldout],np.unique(keys[train])).mean()))
            for name,x in design.items():
                before=time.perf_counter();guided=name.startswith('guided_');params=ridge_fit(x,inverse_arrays if guided else truth_arrays,train,regions=9 if name=='guided_spatial' else 1)
                for style in [0,1]:
                    coefficients=params[:,style*3:style*3+3]
                    prediction=post_forward(x@coefficients,features['post_tables']).reshape(rgb.shape) if guided else np.rint(np.clip(x@coefficients,0,1)*255).astype(np.uint8).reshape(rgb.shape)
                    native_difference=None
                    if guided:
                        native=render_coefficients(model,api,rgb,coefficients);native_difference=int(np.abs(prediction.astype(int)-native.astype(int)).max())
                        if native_difference>1:raise ValueError('Design and actual native renderer diverge: '+str((rows[style]['index'],name,native_difference)))
                        prediction=native
                    outputs[style][name].reshape(-1,3)[heldout]=prediction.reshape(-1,3)[heldout]
                    details[style][name].append(dict(fold=fold,train=errors(prediction,truth[style],train),heldout=errors(prediction,truth[style],heldout),maximum_coefficient=float(np.abs(coefficients).max()),native_max_code_difference=native_difference))
                seconds=time.perf_counter()-before
                for style in [0,1]:details[style][name][-1]['shared_two_style_fit_render_seconds']=seconds
        for style in [0,1]:
            row=rows[style];path=current/f'{row["index"]}.png'
            if digest(path)!=score_lookup[row['index']]['sha256']:raise ValueError('Changed current prediction')
            with Image.open(path) as im:baseline=np.array(im.convert('RGB'))
            baseline_error=errors(baseline,truth[style])
            if baseline_error['mae']!=score_lookup[row['index']]['mae']:raise ValueError('Current MAE does not reproduce')
            case=dict(index=row['index'],group_id=group,style_index=style,current=baseline_error,models={name:dict(**errors(prediction,truth[style]),folds=details[style][name]) for name,prediction in outputs[style].items()},
                collision_floor=rgb_collision_floor(rgb,truth[style]),post_range=inverses[style][1],guide=features['guide'],post_min_codes=(features['post_tables'][0]*255).tolist(),exact_train_rgb_coverage_by_fold=coverage)
            report['cases'].append(case)
            for name,prediction in outputs[style].items():Image.fromarray(prediction).save(out/f'{row["index"]}-{name}.png')
            canvas=Image.new('RGB',(1024,550),'#222');draw=ImageDraw.Draw(canvas)
            panels=[('Target',rgb),('GT',truth[style]),('Current 13 (no GT fit)',baseline),('GT-fitted affine',outputs[style]['affine']),('GT-fitted quadratic',outputs[style]['quadratic']),('GT-fitted guided global',outputs[style]['guided_global']),('GT-fitted guided spatial',outputs[style]['guided_spatial'])]
            for i,(label,image) in enumerate(panels):
                x=i%4*256;y=i//4*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(image),(x,y+19))
            draw.text((771,300),'Two spatial folds; GT-assisted audit only',fill='white');canvas.save(out/f'{row["index"]}.jpg',quality=95)
        (out/'progress.json').write_text(json.dumps(dict(groups=number+1,cases=len(report['cases']))),encoding='utf-8')
        print(json.dumps(dict(groups=number+1,cases=len(report['cases']))),flush=True)
    if len(report['cases'])!=64 or digest(checkpoint)!=checkpoint_sha or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(score_path)!=report['scores_sha256']:raise ValueError('Incomplete or changed evidence')
    report['summary']={}
    for name in design:
        values=[r['models'][name]['mae'] for r in report['cases']];report['summary'][name]=dict(mean_mae=float(np.mean(values)),median_mae=float(np.median(values)),at_most_3=sum(v<=3 for v in values),at_most_5=sum(v<=5 for v in values))
    differences=[r['models']['guided_global']['mae']-r['models']['guided_spatial']['mae'] for r in report['cases']]
    report.update(finished=True,current_mean=float(np.mean([r['current']['mae'] for r in report['cases']])),spatial_gain=dict(mean=float(np.mean(differences)),at_least_one_code=sum(v>=1 for v in differences),regressed_over_point_one=sum(v<-.1 for v in differences)),
        mean_rgb_collision_floor=float(np.mean([r['collision_floor']['minimum_mae'] for r in report['cases']])),wall_seconds=time.perf_counter()-started,app_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ['finished','summary','current_mean','spatial_gain','mean_rgb_collision_floor','wall_seconds']}),flush=True)


if __name__=='__main__':main()
