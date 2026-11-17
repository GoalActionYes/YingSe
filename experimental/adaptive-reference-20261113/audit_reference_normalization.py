"""Verify label provenance/counts/distributions, not training or quality."""
import json
from pathlib import Path
import numpy as np
from independent_coco import ROOT,digest


def main():
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';out=base/'reference-normalization-inputs-v1';load=lambda p:json.loads(p.read_text())
    r=load(out/'results.json');assert r['finished'] and r['model_state_before']==r['model_state_after']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=base/'cohort-training-coverage-v2';assert digest(data/'cohort.json')==r['cohort_sha256'] and digest(data/'srgb-v1/manifest.json')==r['prepared_manifest_sha256']
    cohort=load(data/'cohort.json');rows={e['index']:e for e in cohort['groups']['train']};contents={e['content'] for e in rows.values()}
    lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']}
    pair_path=base/'reference-original-coverage-v1/cohort-training-coverage-v2.json';assert digest(pair_path)==r['pair_report_sha256'];pairs=load(pair_path)['selected']
    expected={index:(original,ref) for index,original,ref in pairs};assert len(expected)==865
    assert len(r['inputs'])==1267 and len(r['labels'])==865
    assert digest(out/'histograms.npy')==r['histograms_sha256'];h=np.load(out/'histograms.npy',allow_pickle=False)
    assert h.shape==(1267,3,128) and h.dtype==np.float32 and np.isfinite(h).all() and h.min()>=0 and np.allclose(h.sum(-1),1,atol=1e-6)
    for e in r['inputs']:
        assert e['parent_sha256']==lookup[e['path']]['sha256'] and digest(data/lookup[e['path']]['prepared_path'])==e['parent_sha256']
    seen=set();originals=set();identities=set();pair_keys=set()
    for e in r['labels']:
        i=e['index'];assert i in rows and i not in seen;seen.add(i)
        original=r['inputs'][e['original_teacher']]['path'];ref=r['inputs'][e['identity_teacher']]['path']
        assert (original,ref)==expected[i] and original in contents and ref==rows[i]['reference']
        originals.add(original);identities.add(ref);pair_keys.add((original,ref))
    assert len(originals)==438 and len(pair_keys)==829 and seen==set(expected)
    assert {e['index'] for e in r['bridge']}=={e['index'] for e in cohort['groups']['validation']} and len(r['bridge'])==64
    assert r['shared_encoder_auxiliary_gradient_l1']>0
    evidence=dict(scope=r['scope'],teacher_input_count=len(r['inputs']),eligible_training_rows=865,unique_originals=len(originals),unique_reference_pairs=len(pair_keys),
        no_teacher_original_outside_training=True,distribution_shape=list(h.shape),histograms_sha256=r['histograms_sha256'],
        inference_bridge_exact_cases=64,shared_encoder_auxiliary_gradient_l1=r['shared_encoder_auxiliary_gradient_l1'],
        model_state_unchanged=True,checkpoint_sha256=r['checkpoint_sha256'],results_sha256=digest(out/'results.json'),
        source_sha256=r['source_sha256'],auditor_sha256=digest(Path(__file__)),training=False,product_integrated=False)
    (ROOT/'docs/research/2026-11-17-reference-normalization-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence))


if __name__=='__main__':main()
