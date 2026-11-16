"""Evaluate fixed selected heads on original train/development, no updates."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from regional_correction import RegionalCorrection,global_control


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=b/'cohort-multiref-v1';cache=b/'regional-correction-inputs-v1';train=b/'regional-correction-training-v1'
    r=json.loads((train/'results.json').read_text());registry=json.loads((cache/'results.json').read_text());cohort=json.loads((data/'cohort.json').read_text())
    assert r['finished'] and digest(cache/'results.json')==r['input_report_sha256']
    assert digest(data/'cohort.json')==registry['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==registry['prepared_manifest_sha256']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    original_path=ROOT/'.tools/TST100K-index/triplet.json';assert digest(original_path)==cohort['index_sha256'];original=json.loads(original_path.read_text())
    lookup={x['original_path']:x for x in json.loads((data/'srgb-v1/manifest.json').read_text())['files']};inputs={x['index']:x for x in registry['inputs']}
    assert digest(cache/'descriptors.npz')==registry['descriptors_sha256']
    with np.load(cache/'descriptors.npz') as a:features=a['regional']
    out=b/'regional-generalization-v1';out.mkdir(exist_ok=False)
    report=dict(scope='Fixed selected checkpoint fit diagnostic, consumed train/development only',training_results_sha256=digest(train/'results.json'),index_sha256=digest(original_path),source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['probe_regional_generalization.py','regional-generalization-protocol.md']},cases=[],finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    engine=SpatialEngine('multiref-v1');heads={}
    for arm in ['control','regional']:
        epoch=r['arms'][arm]['best_epoch'];p=train/arm/f'epoch-{epoch}.pth';assert digest(p)==r['arms'][arm]['epochs'][epoch-1]['checkpoint_sha256']
        h=RegionalCorrection().to(engine.device);h.load_state_dict(torch.load(p,map_location=engine.device,weights_only=True));h.eval().requires_grad_(False);heads[arm]=h
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for split in ['train','validation']:
        rows=[x for x in registry['rows'] if x['split']==split];assert len(rows)==len(cohort['groups'][split])
        with torch.inference_mode():
            for i,(row,item) in enumerate(zip(rows,cohort['groups'][split])):
                assert all(row[k]==item[k] for k in ['index','group_id','style_index'])
                assert all(item[k]==original[item['index']][k] for k in ['content','reference','gt'])
                a,z=[inputs[row['roles'][role]['feature_index']] for role in ['content','reference']]
                assert all(row['roles'][role]['original_path']==item[role] for role in ['content','reference'])
                rgb,ref=[image(ROOT/e['path'],e['parent_sha256']) for e in [a,z]]
                raw=engine.prepare(rgb,ref).float_output(rgb);rawtensor=torch.from_numpy(raw).permute(2,0,1)[None].to(engine.device)
                assert digest(cache/a['mask_path'])==a['mask_sha256'];prob=torch.from_numpy(np.load(cache/a['mask_path'])[None,None]).to(engine.device)
                c,s=[torch.from_numpy(features[e['index']][None]).to(engine.device) for e in [a,z]];preds={'baseline':np.rint(raw.clip(0,1)*255).astype(np.uint8)}
                for arm,h in heads.items():
                    cc,ss=(global_control(c),global_control(s)) if arm=='control' else (c,s)
                    actual=h(engine.tensor(rgb),rawtensor,prob,cc,ss);preds[arm]=np.rint(actual[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
                    if split=='validation':
                        epoch=r['arms'][arm]['best_epoch'];p=train/arm/f'epoch-{epoch}'/f"{item['index']}.png"
                        with Image.open(p) as im:assert np.array_equal(preds[arm],np.array(im.convert('RGB')))
                e=lookup[item['gt']];gt=image(data/e['prepared_path'],e['sha256']).astype(float)
                supported=all(.02<=v['soft_person_fraction']<=.98 and .02<=v['hard_person_fraction']<=.98 for v in [a,z])
                record=dict(split=split,index=item['index'],supported=supported,methods={m:dict(mae=float(np.abs(p.astype(float)-gt).mean()),pixel_sha256=hashlib.sha256(p.tobytes()).hexdigest(),change_from_baseline=float(np.abs(p.astype(float)-preds['baseline']).mean())) for m,p in preds.items()})
                report['cases'].append(record)
                if (i+1)%128==0:print(json.dumps(dict(split=split,evaluated=i+1)),flush=True)
    report['summary']={}
    for split in ['train','validation']:
        rows=[x for x in report['cases'] if x['split']==split];summary={}
        for arm in ['control','regional']:
            baseline=np.array([x['methods']['baseline']['mae'] for x in rows]);a=np.array([x['methods'][arm]['mae'] for x in rows])
            summary[arm]=dict(baseline_mean=float(baseline.mean()),mean=float(a.mean()),improvement_percent=float((1-a.mean()/baseline.mean())*100),improved=int((baseline-a>.1).sum()),severe=int(((a-baseline>5)&(a-baseline>.25*baseline)).sum()))
        report['summary'][split]=summary
    assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2))
    (ROOT/'docs/research/2026-11-17-regional-generalization-evidence.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
