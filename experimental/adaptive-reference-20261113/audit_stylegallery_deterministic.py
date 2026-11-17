"""Check persisted deterministic evidence, source identity and timeout boundary."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from screen_stylegallery import ROOT,BASE,digest


def main():
    load=lambda p:json.loads(Path(p).read_text());gradient_path=ROOT/'docs/research/2026-11-17-stylegallery-deterministic-evidence.json';g=load(gradient_path)
    assert g['passed'] and g['deterministic_algorithms'] and not g['warn_only'] and g['cudnn_deterministic'] and not g['cudnn_benchmark'] and g['cublas_workspace_config']==':4096:8'
    assert len(g['trials'])==3 and len({e['gradient_sha256'] for e in g['trials']})==1 and len({e['loss'] for e in g['trials']})==1
    assert all(e['loss_exact'] and e['gradient_exact'] and e['maximum_gradient_difference']==0 and e['mean_gradient_difference']==0 for e in g['comparisons'].values())
    for n,h in g['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    out=BASE/'stylegallery-screen-deterministic-baseline-v1';r=load(out/'results.json');frozen=load(out/'frozen-config.json');s=load(out/'22927/status.json')
    assert r['stop_reason']=='child_timeout' and r['failed_index']==22927 and not r['completed'] and not r['measurements'] and not r['finished'] and 600<=r['elapsed_seconds']<610
    assert r['strict_deterministic'] and r['controlled_single_case'] and not r['activation_checkpoint'] and r['expected_count']==1
    assert s['strict_deterministic'] and not s['warn_only'] and s['cublas_workspace_config']==':4096:8' and s['stage']=='style_transfer'
    assert sum(e['state_tensors_verified'] for e in s['components'].values())==1130 and all(e['initial_values_exact'] for e in s['components'].values())
    assert not (out/'22927/native512.png').exists() and not (out/'22927/metric256.png').exists()
    for n,h in r['source_sha256'].items():assert digest(out/'frozen-source'/n)==digest(Path(__file__).with_name(n))==h
    verified=0
    for row in frozen['rows']:
        for e in row['inputs'].values():
            path=out/e['path'];assert digest(path)==e['sha256']
            with Image.open(path) as im:a=np.array(im)
            import hashlib
            assert hashlib.sha256(a.tobytes()).hexdigest()==e['pixel_sha256'];verified+=1
    old=load(BASE/'stylegallery-screen-checkpoint-bridge-v1/results.json');assert not old['bridge']['native_exact'] and not old['bridge']['metric_exact']
    evidence=dict(scope='Persisted record/source/input audit only; no independent GPU re-execution',gradient_evidence_sha256=digest(gradient_path),gradient_records_consistent=True,full_image_results_sha256=digest(out/'results.json'),frozen_config_sha256=digest(out/'frozen-config.json'),inputs_verified=verified,initial_state_tensors_verified_in_runtime=1130,full_image_completed=False,stop_reason=r['stop_reason'],elapsed_seconds=r['elapsed_seconds'],load_seconds=s['load_seconds'],cluster_seconds=s['cluster_seconds'],cluster_peak_cuda_allocated_bytes=s['cluster_peak_cuda_allocated_bytes'],old_failed_bridge_preserved=True,auditor_sha256=digest(Path(__file__)),product_integrated=False)
    (ROOT/'docs/research/2026-11-17-stylegallery-deterministic-audit-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(evidence))


if __name__=='__main__':main()
