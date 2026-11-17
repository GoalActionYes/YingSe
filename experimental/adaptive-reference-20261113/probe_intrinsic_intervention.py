"""Frozen spatial lighting-prior assay; labels only used after image inference."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from probe_intrinsic_prior import ROOT,ASSETS,digest,to_linear

SOURCE=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
FIRST=ROOT/'outputs/adaptive-reference-20261113/intrinsic-prior-first-v1'
OUT=ROOT/'outputs/adaptive-reference-20261113/intrinsic-intervention-v1'

def log_color(rgb):
    a=np.maximum(rgb.astype(np.float64),1e-4)
    return np.stack((np.log(a[...,0]/a[...,1]),np.log(a[...,2]/a[...,1])),-1)

def describe(array,baseline,mask):
    if not mask.any():return None
    vector=(log_color(array)-log_color(baseline))[mask].mean(0)
    return dict(vector=vector.tolist(),norm=float(np.linalg.norm(vector)),
                floor_fraction=float((array[mask]<1e-4).mean()),pixels=int(mask.sum()))

def checks(rows,supported):
    by={r['name']:r for r in rows};out=[]
    for color in ['green','magenta']:
        whole=by['whole_'+color]['regions']['primary'];bg=by['background_'+color+'_100']['regions']['primary']
        if not supported or whole is None or bg is None:
            out.append(dict(color=color,supported=False,passed=False));continue
        v=np.array(whole['shading']['vector']);k=np.array(whole['input']['vector']);n=np.linalg.norm(v);kn=np.linalg.norm(k)
        cosine=float(v@k/(n*kn)) if n*kn else None;fraction=float(n/kn) if kn else None
        ratio=float(bg['shading']['norm']/n) if n else None
        floors=[by['identity']['regions']['primary']['shading']['floor_fraction'],whole['shading']['floor_fraction'],bg['shading']['floor_fraction']]
        response=cosine is not None and cosine>=.5 and fraction>=.25
        stable=ratio is not None and ratio<=.5;reliable=max(floors)<=.01
        out.append(dict(color=color,supported=True,cosine=cosine,response_fraction=fraction,background_to_whole_ratio=ratio,
            response_pass=bool(response),invariance_pass=bool(stable),floor_reliable=bool(reliable),passed=bool(response and stable and reliable)))
    return out

def main():
    OUT.mkdir(exist_ok=False)
    for p in [ROOT/'.tools/DeepWB-audit/vendor',ROOT/'.tools/StyleGallery-audit/vendor']:sys.path.insert(0,str(p))
    import torch,diffusers
    from diffusers import MarigoldIntrinsicsPipeline
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    first=json.loads((FIRST/'results.json').read_text());assert first['finished'] and first['exact_repeat']
    manifest=json.loads((ASSETS/'assets.json').read_text());assert manifest==first['assets']
    for f in manifest['files']:assert digest(ASSETS/'models'/f['path'])==f['sha256']
    implementation=Path(diffusers.__file__).parent/'pipelines/marigold'
    assert all(digest(implementation/n)==h for n,h in first['implementation_sha256'].items())
    parent=json.loads((SOURCE/'results.json').read_text());assert parent['finished'] and len(parent['cases'])==6
    names=['probe_intrinsic_intervention.py','intrinsic-intervention-protocol.md','probe_intrinsic_prior.py']
    report=dict(scope='controlled spatial lighting estimate, not transfer quality or physical truth',assets=manifest,
        source_results_sha256=digest(SOURCE/'results.json'),first_results_sha256=digest(FIRST/'results.json'),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},config=first['config'],cases=[],finished=False,product_integrated=False)
    for n in names:(OUT/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    start=time.perf_counter();pipe=MarigoldIntrinsicsPipeline.from_pretrained(ASSETS/'models',variant='fp16',
        torch_dtype=torch.float16,use_safetensors=True,low_cpu_mem_usage=True,local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True);torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-start
    def infer(rgb,path):
        assert time.perf_counter()-start<600
        t=time.perf_counter()
        with torch.inference_mode():pred=pipe(Image.fromarray(rgb),num_inference_steps=4,ensemble_size=1,processing_resolution=768,
            batch_size=1,generator=torch.Generator(device='cuda').manual_seed(20261117),output_type='np').prediction
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        assert seconds<=60 and pred.shape==(3,256,256,3) and np.isfinite(pred).all()
        np.save(path,pred,allow_pickle=False)
        return pred,dict(file=str(path.relative_to(OUT)),sha256=digest(path),seconds=seconds)
    rgb=np.array(Image.open(FIRST/'input.png'));bridge,record=infer(rgb,OUT/'bridge.npy')
    assert np.array_equal(bridge,np.load(FIRST/first['runs'][0]['file'],allow_pickle=False));report['bridge']=record
    for case in parent['cases']:
        folder=OUT/case['pair'];folder.mkdir();pending=[];identity=None;base=None
        for row in case['variants']:
            path=SOURCE/row['reference']['path'];assert digest(path)==row['reference']['sha256']
            im=Image.open(path);assert im.mode=='RGB' and im.size==(256,256);rgb=np.array(im)
            prediction,record=infer(rgb,folder/(row['name']+'.npy'))
            if row['name']=='identity':identity=rgb.copy();base=prediction.copy()
            elif row['name']=='identity_repeat':assert np.array_equal(base,prediction)
            pending.append((row['name'],rgb,prediction,record,row['reference']['sha256']))
            print(json.dumps(dict(pair=case['pair'],variant=row['name'],seconds=round(record['seconds'],3))),flush=True)
        # Offline labels are first accessed after all predictions for this reference have been saved.
        mask_path=SOURCE/case['pair']/'reference-mask.png';person=np.array(Image.open(mask_path).convert('L'))>0
        primary=person&((identity>=32)&(identity<=223)).all(-1);descriptive=not person.any()
        regions=dict(primary=primary,person=person,all=np.ones(person.shape,bool));rows=[]
        for name,rgb,prediction,record,input_sha in pending:
            stats={key:None if not mask.any() else dict(shading=describe(prediction[1],base[1],mask),
                    albedo=describe(prediction[0],base[0],mask),input=describe(to_linear(rgb/255),to_linear(identity/255),mask)) for key,mask in regions.items()}
            rows.append(dict(name=name,input_sha256=input_sha,prediction=record,regions=stats))
        report['cases'].append(dict(pair=case['pair'],mask_sha256=digest(mask_path),primary_pixels=int(primary.sum()),
            person_pixels=int(person.sum()),descriptive_only=descriptive,rows=rows,checks=checks(rows,int(primary.sum())>=32)))
        (OUT/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    report.update(finished=True,seconds=time.perf_counter()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  necessary_conditions_pass=all(x['passed'] for c in report['cases'] if not c['descriptive_only'] for x in c['checks']))
    assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(finished=True,seconds=report['seconds'],necessary_conditions_pass=report['necessary_conditions_pass'])),flush=True)

if __name__=='__main__':main()
