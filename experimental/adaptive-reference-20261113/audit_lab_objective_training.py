"""Recompute every persisted dev score/gate and verify budget/order/state identity."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from coverage_training import block_batches
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'lab-objective-training-v1';load=lambda p:json.loads(p.read_text())
    r=load(out/'results.json');assert r['finished'] and all(v['finished'] for v in r['arms'].values())
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==digest(out/'frozen-source'/n)==h
    for n,h in r['protected_sha256'].items():assert digest(base/n)==h
    data=base/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');dev=cohort['groups']['validation'];train=cohort['groups']['train']
    lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']};truth={}
    for row in dev:
        e=lookup[row['gt']];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:truth[row['index']]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    warm=base/'training-coverage-expanded-v2';initial=torch.load(warm/'epoch-4.pth',map_location='cpu',weights_only=True)
    baseline=load(warm/'results.json')['baseline'];start=load(warm/'epoch-4/scores.json');original=load(base/'appearance-pairwise-replica-v1/epoch-7/scores.json')
    records={};order_lists={};all_scores={};verified=0
    for arm in ['rgb_control','lab_candidate']:
        record=r['arms'][arm];folder=out/arm;rng=np.random.default_rng(r['config']['seed']);means=[];orders=[];scores_by_epoch=[]
        assert record['initial_model_state_exact'] and record['initial_full_scores_exact']
        assert len(record['epochs'])==4
        for epoch in range(5):
            checkpoint=folder/f'epoch-{epoch}.pth';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state.keys()==initial.keys() and all(v.shape==initial[k].shape and torch.isfinite(v).all() for k,v in state.items())
            if epoch==0:assert all(torch.equal(v,initial[k]) for k,v in state.items())
            saved=load(folder/f'epoch-{epoch}/scores.json');assert len(saved['cases'])==64
            predictions={}
            for e in saved['cases']:
                p=folder/f'epoch-{epoch}'/f"{e['index']}.png";assert digest(p)==e['sha256']
                with Image.open(p) as im:a=np.array(im)
                assert a.shape==(256,256,3) and a.dtype==np.uint8;predictions[e['index']]=a;verified+=1
            actual=score_outputs(dev,predictions,truth)
            for e in actual['cases']:e['sha256']=digest(folder/f'epoch-{epoch}'/f"{e['index']}.png")
            assert actual==saved;means.append(actual['mean']);scores_by_epoch.append(actual)
            if epoch==0:
                assert actual['mean']==start['mean'] and actual['paired']==start['paired']
                for row in dev:
                    with Image.open(warm/'epoch-4'/f"{row['index']}.png") as im:assert np.array_equal(predictions[row['index']],np.array(im))
            else:
                e=record['epochs'][epoch-1];order=[train[i]['index'] for batch in block_batches(train,rng,1) for i in batch]
                h=hashlib.sha256(json.dumps(order).encode()).hexdigest();orders.append(h)
                assert len(order)==len(set(order))==4096 and h==e['order_sha256'] and e['updates']==epoch*1024
                assert e['mean']==actual['mean'] and e['delta_mean']==actual['delta_mean'] and digest(checkpoint)==e['checkpoint_sha256']
                assert np.isfinite(e['loss_means']).all()
        best=int(np.argmin(means));assert best==record['best_epoch'] and means[best]==record['best_mean']
        assert digest(folder/f'epoch-{best}.pth')==record['selected_checkpoint_sha256']
        order_lists[arm]=orders;all_scores[arm]=scores_by_epoch[best]
        records[arm]=dict(best_epoch=best,best_mean=means[best],updates=4096,png_verified=320,initial_state_exact=True,all_scores_recomputed=True,
            epoch_summary=[{k:e[k] for k in ['epoch','updates','mean','delta_mean','loss_means','order_sha256']} for e in record['epochs']])
    assert order_lists['rgb_control']==order_lists['lab_candidate']
    for arm in records:
        comparisons={'original_C10':original,'expanded_start':start}
        if arm=='lab_candidate':comparisons['matched_control']=all_scores['rgb_control']
        gates={name:response_development_gate(baseline,all_scores[arm],c) for name,c in comparisons.items()}
        assert gates==r['arms'][arm]['gates']
        passed=bool(arm=='lab_candidate' and all(g['proceed_to_independent_evaluation'] for g in gates.values()))
        assert passed==r['arms'][arm]['full_candidate_gate'];records[arm].update(gates=gates,full_candidate_gate=passed)
    selected_states=[torch.load(out/arm/f"epoch-{r['arms'][arm]['best_epoch']}.pth",map_location='cpu',weights_only=True) for arm in ['rgb_control','lab_candidate']]
    encoder_keys=[k for k in selected_states[0] if k.startswith('img_projection.')]
    encoder_comparison=dict(tensors=len(encoder_keys),unequal_tensors=sum(not torch.equal(selected_states[0][k],selected_states[1][k]) for k in encoder_keys),
        maximum_absolute_difference=max(float((selected_states[0][k]-selected_states[1][k]).abs().max()) for k in encoder_keys))
    evidence=dict(scope=r['scope'],config=r['config'],results_sha256=digest(out/'results.json'),source_sha256=r['source_sha256'],auditor_sha256=digest(Path(__file__)),
        arms=records,total_prediction_pngs_verified=verified,training_orders_identical=True,product_integrated=False)
    evidence['selected_shared_encoder_comparison']=encoder_comparison
    (ROOT/'docs/research/2026-11-17-lab-objective-training-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(verified=verified,arms={k:{a:v[a] for a in ['best_epoch','best_mean','full_candidate_gate']} for k,v in records.items()})))


if __name__=='__main__':main()
