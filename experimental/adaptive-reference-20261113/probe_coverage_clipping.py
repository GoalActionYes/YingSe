"""Frozen eval-only raw-output clipping statistics with a64PNG native bridge."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from clipping_diagnostic import clipping_statistics


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';trial=base/'training-coverage-expanded-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(trial/'results.json');prep=load(data/'srgb-v1/manifest.json');rows=load(data/'cohort.json')['groups']['validation'];geometry=load(base/'coverage-response-geometry-v1/results.json');strata={v['group_id']:v['stratum'] for v in geometry['cases']};weight=trial/'epoch-4.pth'
    if not r['finished'] or r['best_epoch']!=4 or digest(weight)!=r['selected_checkpoint_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Frozen trial changed')
    lookup={v['original_path']:v for v in prep['files']};cache={}
    for name in {v[k] for v in rows for k in ('content','reference','gt')}:
        p=data/lookup[name]['prepared_path']
        if digest(p)!=lookup[name]['sha256']:raise ValueError('Prepared input changed')
        with Image.open(p) as im:cache[name]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);tensor=lambda x:(torch.from_numpy(x.copy()).permute(2,0,1)[None].float()/255).cuda();cases=[]
    with torch.inference_mode():
        for row in rows:
            raw=appearance_forward(model,api,tensor(cache[row['content']]),tensor(cache[row['reference']]))[0][0].cpu().permute(1,2,0).numpy();rgb=np.rint(raw.clip(0,1)*255).astype(np.uint8)
            with Image.open(trial/f'epoch-4/{row["index"]}.png') as im:saved=np.array(im.convert('RGB'))
            if not np.array_equal(rgb,saved):raise ValueError('Native selected RGB bridge differs')
            cases.append(dict(index=row['index'],group_id=row['group_id'],stratum=strata[row['group_id']],**clipping_statistics(raw,cache[row['gt']]/255.)))
    summaries={s:{k:float(np.mean([v[k] for v in cases if s=='all' or v['stratum']==s])) for k in ('outside_fraction','outside_error_fraction','clipped_l1','raw_l1')} for s in ('all','improved','regressed','near_zero')}
    if digest(weight)!=r['selected_checkpoint_sha256']:raise ValueError('Weights changed')
    out=base/'coverage-clipping-diagnostic-v1';out.mkdir(exist_ok=False);names=['probe_coverage_clipping.py','clipping_diagnostic.py','clipping-diagnostic-protocol.md'];frozen=out/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    report=dict(scope='Frozen eval snapshot only, not training-gradient history or causal/semantic proof',source_sha256={n:digest(frozen/n) for n in names},trial_results_sha256=digest(trial/'results.json'),weight_sha256=digest(weight),prepared_manifest_sha256=digest(data/'srgb-v1/manifest.json'),native_png_bridge=64,cases=cases,summaries=summaries,finished=True,training_started=False,app_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
