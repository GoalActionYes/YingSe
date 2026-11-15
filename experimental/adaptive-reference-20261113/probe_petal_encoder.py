"""Pre-frozen consumed-train correspondence diagnostic, no optimizer or training."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from independent_coco import ROOT,digest
from petal_model import PetalModel,tensor
from petal_encoder import encode
from real_style_correspondence import correspondence


def main():
    torch.set_num_threads(4)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    data=base/'cohort-multiref-v1'
    old_dir=base/'frozen-grade-correspondence-v1'
    old_path=old_dir/'results.json'
    read_json=lambda p:json.loads(p.read_text(encoding='utf-8'))
    prior=read_json(old_path)
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json'
    if not prior['finished'] or digest(cohort_path)!=prior['cohort_sha256'] or digest(prep_path)!=prior['prepared_manifest_sha256'] or digest(old_dir/'features.npz')!=prior['features_sha256']:
        raise ValueError('Historical baseline evidence mismatch')
    rows=read_json(cohort_path)['groups']['train'];lookup={v['original_path']:v for v in read_json(prep_path)['files']}
    groups={}
    for i,row in enumerate(rows):groups.setdefault(row['group_id'],[]).append(i)
    if len(rows)!=1024 or len(groups)!=256 or any(sorted(rows[i]['style_index'] for i in g)!=list(range(4)) or len({rows[i]['content'] for i in g})!=1 for g in groups.values()):raise ValueError('Unexpected train grouping')
    pixels=[];records=[]
    for row in rows:
        for key in ['reference','gt']:
            relative=row[key];item=lookup[relative];path=data/item['prepared_path']
            if digest(path)!=item['sha256']:raise ValueError('Changed source file')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            record=dict(path=relative,parent_sha256=item['sha256'],pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            if record!=prior['pixel_records'][len(records)]:raise ValueError('Historical pixels/order differ')
            # Keep only file records; 2048 full images would consume unnecessary RAM.
            records.append(record);pixels.append(path)
    out=base/'petal-encoder-correspondence-v1';out.mkdir(exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['probe_petal_encoder.py','petal_encoder.py','petal_model.py','real_style_correspondence.py','independent_coco.py','spatial_render.py','petal-encoder-protocol.md','test_petal_encoder.py']:
        (frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Consumed training-pair frozen PETAL encoder diagnostic; H128 training advantage and unknown PETAL pretraining overlap',source_sha256={p.name:digest(p) for p in frozen.iterdir()},prior_results_sha256=digest(old_path),baseline_features_sha256=prior['features_sha256'],cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),pixel_records=records,finished=False,training_started=False,app_promoted=False,actual_independent_eligibility=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model=PetalModel();report['provenance']=model.provenance;before=model.state_digest();report['model_state_before_sha256']=before
    def read(index):
        path=pixels[index]
        if digest(path)!=records[index]['parent_sha256']:raise ValueError('Input changed during diagnostic')
        with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        if hashlib.sha256(rgb.tobytes()).hexdigest()!=records[index]['pixel_sha256']:raise ValueError('Pixel conversion changed')
        return rgb
    pair=[read(0),read(1)];captured=[]
    handle=model.style.register_forward_hook(lambda _m,_i,o:captured.append(F.normalize(o['z'],dim=-1).detach().clone()))
    try:
        with torch.inference_mode(),torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            model.api.inference(model.style,model.transfer,tensor(pair[1],model.device),tensor(pair[0],model.device),guidance_scale=1.,style_input_size=512)
    finally:handle.remove()
    bridge=encode(model,pair)
    if len(captured)!=1 or not torch.equal(bridge,captured[0]):raise ValueError('Official encoder extraction bridge mismatch')
    report['bridge']=dict(maximum_float_difference=float((bridge-captured[0]).abs().max()),batch_size=2,dimensions=bridge.shape[1],same_runtime_only=True)
    del captured,bridge,pair
    features=[];seconds=[];torch.cuda.reset_peak_memory_stats()
    for i in range(0,len(pixels),2):
        pair=[read(i),read(i+1)];torch.cuda.synchronize();start=time.perf_counter()
        feature=encode(model,pair).cpu().numpy();torch.cuda.synchronize();seconds.append(time.perf_counter()-start)
        if feature.shape!=(2,1024):raise ValueError('Unexpected public encoder dimensions')
        features.append(feature)
        if (i+2)%128==0:print(json.dumps(dict(images_extracted=i+2,total=2048)),flush=True)
    arrays={'PETAL1024':np.concatenate(features)}
    with np.load(old_dir/'features.npz') as cached:
        for name in ['H128','G64']:arrays[name]=cached[name].copy()
    np.savez_compressed(out/'features.npz',**arrays);report['features_sha256']=digest(out/'features.npz')
    report['model_state_after_sha256']=model.state_digest()
    if before!=report['model_state_after_sha256'] or any(m.training or any(p.requires_grad for p in m.parameters()) for m in [model.style,model.transfer]):raise ValueError('Frozen state changed')
    report['timing']=dict(pair_encoder_seconds_median=float(np.median(seconds)),pair_encoder_seconds_maximum=float(max(seconds)),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),input_file_reads_excluded=True,transfer_network_excluded=True)
    report['families']={}
    for name,array in arrays.items():
        value=array.astype(np.float64);norm=np.linalg.norm(value,axis=1,keepdims=True)
        if value.shape[0]!=2048 or not np.isfinite(value).all() or (norm<=1e-12).any():raise ValueError('Invalid cached features')
        value=value/np.maximum(norm,1e-12);results=[]
        for group_id,indices in groups.items():
            order=sorted(indices,key=lambda i:rows[i]['style_index']);a=value[[i*2 for i in order]];b=value[[i*2+1 for i in order]]
            cost=np.clip(1-a@b.T,0,2);results.append(dict(group_id=group_id,cost=cost.tolist(),**correspondence(cost)))
        summary={k:float(np.mean([r[k] for r in results])) for k in ['row_top1','assignment_fraction_correct','identity_optimum_fraction','diagonal_mean','offdiagonal_mean','margin']}
        report['families'][name]=dict(dimensions=value.shape[1],summary=summary,groups=results)
        if name in ['H128','G64'] and summary!=prior['families'][name]['summary']:raise ValueError('Historical baseline score changed')
        print(json.dumps(dict(family=name,summary=summary)),flush=True)
    h,p=[report['families'][n]['summary'] for n in ['H128','PETAL1024']]
    report['research_value_checks']=dict(top1_at_least_half=p['row_top1']>=.5,top1_gain_at_least_five_points=p['row_top1']>=h['row_top1']+.05,assignment_no_worse=p['assignment_fraction_correct']>=h['assignment_fraction_correct'])
    report['may_design_transfer_comparison']=all(report['research_value_checks'].values())
    if digest(old_path)!=report['prior_results_sha256'] or digest(old_dir/'features.npz')!=report['baseline_features_sha256'] or any(digest(Path(__file__).with_name(n))!=sha for n,sha in report['source_sha256'].items()):raise ValueError('Diagnostic evidence changed')
    report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,research_value_checks=report['research_value_checks'],may_design_transfer_comparison=report['may_design_transfer_comparison'],training_started=False)),flush=True)


if __name__=='__main__':main()
