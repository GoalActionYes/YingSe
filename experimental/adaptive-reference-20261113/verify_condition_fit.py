"""Fresh GPU rerender from stored deltas only; no optimization, no model update."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from petal_condition import condition_forward


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';folder=base/'condition-fit-v2';load=lambda p:json.loads(p.read_text(encoding='utf-8'));report=load(folder/'results.json');data=base/'cohort-multiref-v1';cohort=load(data/'cohort.json');prep=load(data/'srgb-v1/manifest.json');lookup={v['original_path']:v for v in prep['files']}
    if not report['finished'] or digest(data/'cohort.json')!=report['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=report['prepared_manifest_sha256']:raise ValueError('Evidence changed')
    model,api,_=author_model();cp=base/'multiref-control-v1/epoch-6.pth'
    if digest(cp)!=report['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);cases=[]
    for row,record in zip(cohort['groups']['validation'],report['cases']):
        arrays=[]
        for role in ['content','reference','gt']:
            entry=lookup[row[role]];p=data/entry['prepared_path']
            if digest(p)!=entry['sha256'] or entry['sha256']!=report['input_parent_hashes'][row[role]]:raise ValueError('Input changed')
            with Image.open(p) as im:arrays.append(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
        matched='--match-batch-layout' in sys.argv
        c,s=[(torch.from_numpy(v[None]).permute(0,3,1,2) if matched else torch.from_numpy(v).permute(2,0,1)[None]).float().cuda()/255 for v in arrays[:2]]
        cache=folder/record['files']['raw']['path']
        if digest(cache)!=record['files']['raw']['sha256']:raise ValueError('Saved parameters changed')
        snapshots=[]
        with np.load(cache) as saved,torch.no_grad():
            for j,step in enumerate(saved['snapshot_steps']):
                delta=torch.from_numpy(saved['snapshot_deltas'][j])[None].cuda();raw,_=condition_forward(model,lambda _a,_b:delta,api,c,s);a=raw[0].cpu().permute(1,2,0).numpy();rgb=np.rint(a.clip(0,1)*255).astype(np.uint8);mae=float(np.abs(rgb.astype(float)-arrays[2].astype(float)).mean())
                if int(step)!=record['snapshots'][j]['step']:raise ValueError('Snapshot step mismatch')
                entry=dict(step=int(step),mae=mae,logged_mae=record['snapshots'][j]['mae'],mae_difference=abs(mae-record['snapshots'][j]['mae']),rgb_array_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                key='baseline_raw' if step==0 else ('selected_raw' if step==record['selected_step'] else None)
                if key:
                    expected=saved[key];expected_rgb=np.rint(expected.clip(0,1)*255).astype(np.uint8)
                    entry.update(maximum_raw_difference=float(np.abs(a-expected).max()),maximum_rgb8_difference=int(np.abs(rgb.astype(int)-expected_rgb.astype(int)).max()),changed_rgb8_channels=int((rgb!=expected_rgb).sum()))
                snapshots.append(entry)
        cases.append(dict(index=row['index'],snapshots=snapshots))
        if len(cases)%16==0:print(json.dumps(dict(cases_replayed=len(cases))),flush=True)
    all_rows=[v for r in cases for v in r['snapshots']];raw_rows=[v for v in all_rows if 'maximum_raw_difference' in v]
    result=dict(scope='Fresh fixed-delta GPU replay difference measurement; original exact assertions failed, no tolerance-based pass claim',results_sha256=digest(folder/'results.json'),verifier_source_sha256=digest(Path(__file__)),snapshot_count=len(all_rows),all_snapshot_scores_exact=all(v['mae_difference']==0 for v in all_rows),all_baseline_and_selected_raw_exact=all(v['maximum_raw_difference']==0 for v in raw_rows),maximum_snapshot_mae_difference=max(v['mae_difference'] for v in all_rows),maximum_saved_raw_difference=max(v['maximum_raw_difference'] for v in raw_rows),maximum_saved_rgb8_difference=max(v['maximum_rgb8_difference'] for v in raw_rows),cases=cases,finished=True)
    result['matched_batch_layout']=matched
    p=folder/('gpu-replay-layout-matched.json' if matched else 'gpu-replay.json')
    if p.exists():raise ValueError('Refusing overwrite')
    p.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
