"""Fixed selected-weight appearance ablations; no training or promotion."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from audit_consistent_reference import score_outputs
from response_geometry import response_geometry


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2';trial=base/'training-coverage-expanded-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));r=load(trial/'results.json');diag=load(base/'coverage-response-geometry-v1/results.json');prep=load(data/'srgb-v1/manifest.json');rows=load(data/'cohort.json')['groups']['validation'];weight=trial/'epoch-4.pth'
    if not r['finished'] or r['best_epoch']!=4 or digest(weight)!=r['selected_checkpoint_sha256'] or digest(data/'srgb-v1/manifest.json')!=r['prepared_manifest_sha256']:raise ValueError('Selected trial changed')
    for n,sha in r['source_sha256'].items():
        if digest(trial/'frozen-source'/n)!=sha or digest(Path(__file__).with_name(n))!=sha:raise ValueError('Trial source changed')
    lookup={v['original_path']:v for v in prep['files']};cache={}
    for path in {v[k] for v in rows for k in ('content','reference','gt')}:
        p=data/lookup[path]['prepared_path']
        if digest(p)!=lookup[path]['sha256']:raise ValueError('Prepared input changed')
        with Image.open(p) as im:cache[path]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    out=base/'coverage-response-ablation-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir();names=set(r['source_sha256'])|{'probe_coverage_response.py','response_geometry.py','coverage-response-ablation-protocol.md'}
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    sources={p.name:digest(p) for p in frozen.iterdir()};model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True));model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False)
    tensor=lambda a:(torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255).cuda();truth={v['index']:cache[v['gt']] for v in rows};groups={v['group_id']:{} for v in rows};strata={v['group_id']:v['stratum'] for v in diag['cases']};modes={}
    for row in rows:groups[row['group_id']][row['style_index']]=row['index']
    with torch.inference_mode():
        for mode in ('full','appearance_off','appearance_target'):
            directory=out/mode;directory.mkdir();predictions={}
            for row in rows:
                c,s=tensor(cache[row['content']]),tensor(cache[row['reference']]);kwargs=dict(disable_appearance=True) if mode=='appearance_off' else dict(appearance_reference=c) if mode=='appearance_target' else {};raw=appearance_forward(model,api,c,s,**kwargs)[0]
                a=np.rint(raw[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8);i=row['index'];predictions[i]=a;Image.fromarray(a).save(directory/f'{i}.png')
                if mode=='full' and digest(directory/f'{i}.png')!=digest(trial/f'epoch-4/{i}.png'):raise ValueError('Full native PNG bridge differs')
            scores=score_outputs(rows,predictions,truth)
            for v in scores['cases']:v['sha256']=digest(directory/f'{v["index"]}.png')
            if mode=='full' and scores!=load(trial/'epoch-4/scores.json'):raise ValueError('Full score bridge differs')
            geometry={name:response_geometry(predictions[pair[0]],predictions[pair[1]],truth[pair[0]],truth[pair[1]]) for name,pair in groups.items()};modes[mode]=dict(scores=scores,geometry=geometry);(directory/'scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8');print(json.dumps(dict(mode=mode,rgb=scores['mean'],delta=scores['delta_mean'])),flush=True)
    if digest(weight)!=r['selected_checkpoint_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in sources.items()):raise ValueError('Frozen probe changed')
    result=dict(scope='Fixed-weight inference ablations, not retrained controls/causal or semantic proof/product selection',trial_results_sha256=digest(trial/'results.json'),selected_checkpoint_sha256=digest(weight),source_sha256=sources,strata=strata,modes=modes,finished=True,training_started=False,app_promoted=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
