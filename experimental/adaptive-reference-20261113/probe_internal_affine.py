"""Observe pre-internal-clamp affine values with exact full-forward bridges."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_grid,appearance_forward
from internal_affine_stage import internal_affine


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';trial=base/'training-coverage-expanded-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(trial/'results.json');prep=load(data/'srgb-v1/manifest.json');rows=load(data/'cohort.json')['groups']['validation'];geometry=load(base/'coverage-response-geometry-v1/results.json');strata={v['group_id']:v['stratum'] for v in geometry['cases']};weight=trial/'epoch-4.pth'
    if not r['finished'] or r['best_epoch']!=4 or digest(weight)!=r['selected_checkpoint_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Trial changed')
    for n,sha in r['source_sha256'].items():
        if digest(trial/'frozen-source'/n)!=sha or digest(Path(__file__).with_name(n))!=sha:raise ValueError('Native source changed')
    lookup={v['original_path']:v for v in prep['files']};cache={}
    for name in {v[k] for v in rows for k in ('content','reference')}:
        p=data/lookup[name]['prepared_path']
        if digest(p)!=lookup[name]['sha256']:raise ValueError('Input changed')
        with Image.open(p) as im:cache[name]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    model,api,author=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);tensor=lambda x:(torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda();cases=[]
    with torch.inference_mode():
        for row in rows:
            c,s=tensor(cache[row['content']]),tensor(cache[row['reference']]);grid=appearance_grid(model,api,c,s);coeffs=model.sampler(grid,c);stage=internal_affine(model,coeffs,c);rebuilt=model.post_guide(stage.clamp(-1,1));native=model.apply_coeffs(coeffs,c);forward,forward_grid=appearance_forward(model,api,c,s)
            difference=max(float((rebuilt-native).abs().max()),float((rebuilt-forward).abs().max()),float((grid-forward_grid).abs().max()))
            if difference!=0:raise ValueError('Reconstructed internal/full/grid bridge differs')
            array=np.rint(rebuilt[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            with Image.open(trial/f'epoch-4/{row["index"]}.png') as im:saved=np.array(im.convert('RGB'))
            if not np.array_equal(array,saved):raise ValueError('Saved selected PNG bridge differs')
            x=stage.cpu().numpy();excess=np.maximum(np.abs(x)-1,0)
            cases.append(dict(index=row['index'],group_id=row['group_id'],stratum=strata[row['group_id']],maximum_float_difference=difference,outside_fraction=float((np.abs(x)>1).mean()),below_fraction=float((x<-1).mean()),above_fraction=float((x>1).mean()),mean_excess=float(excess.mean()),maximum_excess=float(excess.max()),minimum_affine=float(x.min()),maximum_affine=float(x.max())))
    summaries={}
    for stratum in ('all','improved','regressed','near_zero'):
        values=[v for v in cases if stratum=='all' or v['stratum']==stratum];summaries[stratum]=dict(cases=len(values),**{k:float(np.mean([v[k] for v in values])) for k in ('outside_fraction','below_fraction','above_fraction','mean_excess')},cases_with_outside=sum(v['outside_fraction']>0 for v in values),maximum_excess=max(v['maximum_excess'] for v in values))
    if digest(weight)!=r['selected_checkpoint_sha256']:raise ValueError('Weights changed')
    out=base/'coverage-internal-affine-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=['probe_internal_affine.py','internal_affine_stage.py','internal-stage-protocol.md']
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    result=dict(scope='Frozen eval pre-internal-clamp affine snapshot only, not RGB/GT quality/training history or causal proof',source_sha256={n:digest(frozen/n) for n in names},author_revision=author['revision'],trial_results_sha256=digest(trial/'results.json'),weight_sha256=digest(weight),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),native_raw_grid_png_bridges=64,cases=cases,summaries=summaries,finished=True,training_started=False,app_promoted=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
