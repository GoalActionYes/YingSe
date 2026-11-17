"""Fixed-state, already-consumed training128 comparison; never an acceptance set."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from appearance_replication import initialize_replication_model
from appearance_pairwise import appearance_forward
from prepare_reference_normalization import state_hash
from lab_objective import color_objective


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';load=lambda p:json.loads(p.read_text())
    probe=base/'lab-objective-probe-v1/results.json';previous=load(probe)
    run=base/'lab-objective-training-v1';record=load(run/'results.json');assert record['finished']
    for n,h in record['protected_sha256'].items():assert digest(base/n)==h
    for n,h in record['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=base/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');rows=cohort['groups']['train'][:128]
    assert [r['index'] for r in rows]==[r['index'] for r in previous['cases']]
    lookup={r['original_path']:r for r in load(data/'srgb-v1/manifest.json')['files']}
    checkpoints={'start':base/'training-coverage-expanded-v2/epoch-4.pth','rgb_last':run/'rgb_control/epoch-4.pth','lab_last':run/'lab_candidate/epoch-4.pth'}
    for arm,ck in [('rgb_control',checkpoints['rgb_last']),('lab_candidate',checkpoints['lab_last'])]:assert digest(ck)==record['arms'][arm]['epochs'][-1]['checkpoint_sha256']
    assert digest(checkpoints['start'])==previous['checkpoint_sha256']
    out=base/'lab-objective-fit-v1';out.mkdir(exist_ok=False)
    names=['diagnose_lab_objective_fit.py','lab-objective-fit-protocol.md']
    report=dict(scope='Consumed training128 fixed-state fit diagnostic; no training, no independent acceptance',source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},training_results_sha256=digest(run/'results.json'),probe_sha256=digest(probe),checkpoints={k:digest(v) for k,v in checkpoints.items()},arms={},finished=False,training=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def read(row,key,expected):
        e=lookup[row[key]];p=data/e['prepared_path'];assert p.resolve().is_relative_to(base.resolve()) and digest(p)==e['sha256']
        with Image.open(p) as im:a=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(a.tobytes()).hexdigest()==expected['pixel_sha256'][key]
        return (torch.from_numpy(a).permute(2,0,1)[None].float()/255).cuda(),a
    model,api,_=author_model();initialize_replication_model(model,torch.load(base/'multiref-control-v1/epoch-6.pth',map_location='cpu',weights_only=True))
    truths={}
    for arm,checkpoint in checkpoints.items():
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=state_hash(model)
        folder=out/arm;folder.mkdir();cases=[]
        for row,old in zip(rows,previous['cases']):
            pairs=[read(row,k,old) for k in ['content','reference','gt']];c,s,gt=[p[0] for p in pairs];truths[row['index']]=pairs[2][1]
            with torch.no_grad():
                pred=appearance_forward(model,api,c,s)[0].clamp(0,1);mse=float(F.mse_loss(pred,gt));color=float(color_objective(pred,gt))
            if arm=='start':assert mse==old['rgb_mse'] and color==old['color_loss']
            pixels=np.rint(pred[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8);path=folder/f"{row['index']}.png";Image.fromarray(pixels).save(path)
            cases.append(dict(index=row['index'],rgb_mse=mse,color_loss=color,rgb_mae=float(np.abs(pixels.astype(np.float64)-truths[row['index']]).mean()),sha256=digest(path)))
        assert before==state_hash(model)
        report['arms'][arm]=dict(cases=cases,mean_rgb_mae=float(np.mean([r['rgb_mae'] for r in cases])),mean_rgb_mse=float(np.mean([r['rgb_mse'] for r in cases])),mean_color_loss=float(np.mean([r['color_loss'] for r in cases])),model_state_unchanged=True)
        print(json.dumps(dict(arm=arm,mean_rgb_mae=report['arms'][arm]['mean_rgb_mae'])),flush=True)
    verified=0
    for arm,r in report['arms'].items():
        for case in r['cases']:
            path=out/arm/f"{case['index']}.png";assert digest(path)==case['sha256']
            with Image.open(path) as im:actual=np.array(im)
            assert float(np.abs(actual.astype(np.float64)-truths[case['index']]).mean())==case['rgb_mae'];verified+=1
        if arm!='start':
            delta=np.array([a['rgb_mae']-b['rgb_mae'] for a,b in zip(report['arms']['start']['cases'],r['cases'])])
            r['versus_start']=dict(mean_improvement=float(delta.mean()),improved_over_point1=int((delta>.1).sum()),regressed_over_point1=int((delta<-.1).sum()),largest_improvement=float(delta.max()),largest_regression=float(-delta.min()))
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(finished=True,png_verified=verified)
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-lab-objective-fit-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,png_verified=verified,summary={k:{a:v for a,v in r.items() if a!='cases'} for k,r in report['arms'].items()})))


if __name__=='__main__':main()
