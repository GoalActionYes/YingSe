"""Recompute allocator record conditions; optionally verify the complete image pair."""
import json,argparse
from pathlib import Path
import numpy as np
from PIL import Image
from screen_stylegallery import ROOT,BASE,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image-pair',action='store_true');args=parser.parse_args();load=lambda p:json.loads(Path(p).read_text())
    path=ROOT/'docs/research/2026-11-17-stylegallery-allocator-evidence.json';r=load(path);assert r['finished']
    for n,h in r['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    snapshots={e['stage']:e for e in r['snapshots']};a=snapshots['before_cache_release'];b=snapshots['after_cache_release'];released=a['reserved']-b['reserved']
    assert released==r['reserved_released'] and a['allocated']==b['allocated'] and r['allocated_unchanged']
    assert len(r['trials'])==3 and len({t['loss'] for t in r['trials']})==1 and len({t['gradient_sha256'] for t in r['trials']})==1
    passed=r['loss_exact'] and r['gradient_exact'] and r['state_unchanged'] and r['allocated_unchanged'] and released>=256*1024*1024;assert passed==r['supports_full_trial']
    evidence=dict(scope='Persisted allocator records and source audit; no GPU re-execution',allocator_results_sha256=digest(path),record_conditions_recomputed=True,reserved_released=released,allocated_unchanged=True,full_image_pair_verified=False,auditor_sha256=digest(Path(__file__)),product_integrated=False)
    if args.image_pair:
        runs=[BASE/'stylegallery-screen-cachetrim-baseline-v1',BASE/'stylegallery-screen-cachetrim-checkpoint-v1'];reports=[load(p/'results.json') for p in runs]
        assert all(t['finished'] and t['strict_deterministic'] and t['trim_after_init'] and t['controlled_single_case'] and not t['measurements'] for t in reports)
        assert not reports[0]['activation_checkpoint'] and reports[1]['activation_checkpoint'] and reports[1]['controlled_baseline_results_sha256']==digest(runs[0]/'results.json')
        assert reports[0]['rows']==reports[1]['rows'] and all(reports[0]['source_sha256'][n]==reports[1]['source_sha256'][n] for n in reports[0]['source_sha256'])
        statuses=[]
        for directory,report in zip(runs,reports):
            for n,h in report['source_sha256'].items():assert digest(directory/'frozen-source'/n)==h
            item=report['completed'][0];assert item['index']==22927 and item['strict_deterministic'] and not item['warn_only'] and item['cublas_workspace_config']==':4096:8'
            assert sum(e['state_tensors_verified'] for e in item['components'].values())==1130 and all(e['initial_values_exact'] for e in item['components'].values())
            trim=item['init_cache_release'];assert trim['state_exact'] and trim['after']['allocated']<=trim['before']['allocated'];statuses.append(item)
            for name,key in [('native512.png','native_sha256'),('metric256.png','metric_sha256')]:assert digest(directory/'22927'/name)==item[key]
            with Image.open(directory/'22927/native512.png') as im:expected=np.array(im.resize((256,256),Image.Resampling.BILINEAR))
            with Image.open(directory/'22927/metric256.png') as im:assert np.array_equal(expected,np.array(im))
        assert statuses[0]['init_cache_release']['state_sha256']==statuses[1]['init_cache_release']['state_sha256']
        exact={}
        for name,key in [('native512.png','native_exact'),('metric256.png','metric_exact')]:
            with Image.open(runs[0]/'22927'/name) as im:a=np.array(im)
            with Image.open(runs[1]/'22927'/name) as im:b=np.array(im)
            d=np.abs(a.astype(np.int16)-b.astype(np.int16));exact[key]=bool(np.array_equal(a,b));assert exact[key]==reports[1]['controlled_bridge'][key]
            evidence[name]=dict(exact=exact[key],maximum_difference=int(d.max()),mean_difference=float(d.mean()))
        assert reports[1]['controlled_bridge']['passed']==all(exact.values())
        evidence.update(full_image_pair_verified=True,full_image_pair_passed=all(exact.values()),full_image_results_sha256=[digest(p/'results.json') for p in runs],image_statuses=statuses)
    (ROOT/'docs/research/2026-11-17-stylegallery-allocator-audit-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(evidence))


if __name__=='__main__':main()
