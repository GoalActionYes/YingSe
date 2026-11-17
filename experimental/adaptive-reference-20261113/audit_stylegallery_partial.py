"""Audit stopped partial screen, failed exact bridge and native prep caches."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from screen_stylegallery import ROOT,BASE,digest


def main():
    load=lambda p:json.loads(Path(p).read_text());old=BASE/'stylegallery-screen-v3';bridge=BASE/'stylegallery-screen-checkpoint-bridge-v1'
    r=load(old/'results.json');b=load(bridge/'results.json');assert r['stop_reason']=='child_failure' and len(r['completed'])==2 and r['failed_index']==39532 and b['finished'] and b['bridge_only']
    for folder,record in [(old,r),(bridge,b)]:
        for n,h in record['source_sha256'].items():assert digest(folder/'frozen-source'/n)==h
    data=BASE/'cohort-training-coverage-v2';cohort=load(data/'cohort.json');lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']};rows={e['index']:e for e in cohort['groups']['validation']};baseline=load(BASE/'training-coverage-expanded-v2/results.json')['baseline'];prior={e['index']:e for e in baseline['cases']}
    measurements=[];verified=0
    for item in r['completed']:
        index=item['index'];folder=old/str(index)
        for filename,key in [('native512.png','native_sha256'),('metric256.png','metric_sha256')]:assert digest(folder/filename)==item[key]
        with Image.open(folder/'native512.png') as im:expected=np.array(im.resize((256,256),Image.Resampling.BILINEAR))
        with Image.open(folder/'metric256.png') as im:pixels=np.array(im)
        assert np.array_equal(expected,pixels)
        e=lookup[rows[index]['gt']];path=data/e['prepared_path'];assert digest(path)==e['sha256']
        with Image.open(path) as im:gt=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        path=BASE/'multiref-control-v1/epoch-6'/f'{index}.png';assert digest(path)==prior[index]['sha256']
        with Image.open(path) as im:a=np.array(im)
        mae=float(np.abs(pixels.astype(float)-gt).mean());baseline_mae=float(np.abs(a.astype(float)-gt).mean());assert baseline_mae==prior[index]['mae']
        measurements.append(dict(index=index,mae=mae,baseline_mae=baseline_mae,severe=bool(mae-baseline_mae>5 and mae>baseline_mae*1.25)));verified+=2
    assert measurements==r['measurements'];failure=load(old/'39532/failure.json');assert failure['stage']=='failed' and 'out of memory' in failure['error']
    differences={}
    for filename,key in [('native512.png','native_exact'),('metric256.png','metric_exact')]:
        with Image.open(old/'22927'/filename) as im:a=np.array(im).astype(np.int16)
        with Image.open(bridge/'22927'/filename) as im:c=np.array(im).astype(np.int16)
        d=np.abs(a-c);exact=bool(np.array_equal(a,c));assert exact==b['bridge'][key]
        differences[filename]=dict(exact=exact,maximum_difference=int(d.max()),mean_difference=float(d.mean()),different_fraction=float((d!=0).mean()))
    caches=[]
    for role in ['content','reference']:
        paths=[folder/'processed_data/inputs'/f'22927-{role}.pt' for folder in [old,bridge]]
        # Only self-generated local caches of this task, never a downloaded pickle.
        records=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];fields={}
        for key in ['feature','unet_feature','mask','C15','depth']:
            a,c=[value[key] for value in records]
            if isinstance(a,torch.Tensor):assert torch.equal(a,c);shape=list(a.shape)
            else:assert np.array_equal(a,c);shape=list(a.shape)
            fields[key]=dict(exact=True,shape=shape)
        caches.append(dict(role=role,files_sha256=[digest(p) for p in paths],fields=fields))
    gradients=load(ROOT/'docs/research/2026-11-17-stylegallery-gradient-evidence.json')
    for n,h in gradients['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    evidence=dict(scope='Stopped2-case screen and failed exact execution bridge; incomplete quality acceptance',partial_results_sha256=digest(old/'results.json'),bridge_results_sha256=digest(bridge/'results.json'),measurements=measurements,verified_partial_pngs=verified,bridge_differences=differences,prep_cache_comparisons=caches,gradient_comparisons=gradients['comparisons'],gradient_evidence_sha256=digest(ROOT/'docs/research/2026-11-17-stylegallery-gradient-evidence.json'),failed_case=dict(index=39532,error_type=failure['error_type'],peak_cuda_allocated_bytes=failure['peak_cuda_allocated_bytes']),full8_complete=False,full_execution_bridge_passed=False,legacy_case_merge_allowed=False,auditor_sha256=digest(Path(__file__)),product_integrated=False)
    (ROOT/'docs/research/2026-11-17-stylegallery-partial-audit-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(dict(verified=True,measurements=measurements,bridge=differences,cache_fields_exact=True)))


if __name__=='__main__':main()
