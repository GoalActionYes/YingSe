"""Pinned author ModFlows default screening on already-consumed development64."""
import os,sys,json,time,hashlib,contextlib,io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from audit_consistent_reference import score_outputs
from pairwise_gate import response_development_gate


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assets=ROOT/'.tools/ModFlows-audit';load=lambda p:json.loads(p.read_text())
    source=assets/'source';manifest=load(assets/'source-manifest.json');weight=load(assets/'weight-manifest.json');dependencies=load(assets/'dependency-plan.json')
    assert manifest['commit']=='ca5da1a2742fb6a63e03b7a1245662f47d278b24' and weight['revision']=='f7b7077946ed02d70a410a58b780cdc3b3883635'
    for e in manifest['files']:assert digest(source/e['path'])==e['sha256']
    for e in dependencies:assert digest(assets/'wheels'/e['filename'])==e['sha256']
    checkpoint=assets/weight['name'];assert checkpoint.stat().st_size==weight['size'] and digest(checkpoint)==weight['sha256']
    sys.path[:0]=[str(assets/'vendor'),str(source)];os.environ['MPLCONFIGDIR']=str(assets/'mpl-cache');os.environ['MPLBACKEND']='Agg'
    from src.encoder import Encoder,enc_preprocess
    from src.inference import run_inference
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-training-coverage-v2'
    assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    assert digest(data/'srgb-v1/manifest.json')=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    cohort=load(data/'cohort.json');rows=cohort['groups']['validation'];assert len(rows)==64
    lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']}
    baseline_record=load(base/'training-coverage-expanded-v2/results.json')['baseline'];old=base/'multiref-control-v1/epoch-6'
    out=base/'modflows-screen-v1';out.mkdir(exist_ok=False);(out/'inputs').mkdir();(out/'candidate').mkdir()
    names=['screen_modflows.py','modflows-screen-protocol.md']
    report=dict(scope='Consumed development64, pinned default ModFlows; no independent acceptance',author_source=manifest,weight=weight,dependencies=dependencies,
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},cohort_sha256=digest(data/'cohort.json'),manifest_sha256=digest(data/'srgb-v1/manifest.json'),
        config=dict(encoder='B6',hidden=1024,output=8195,enc_steps=8,strength=1.,crop=False,compress=False),rows=[],finished=False,training=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def read(relative):
        e=lookup[relative];p=data/e['prepared_path'];assert p.resolve().is_relative_to(base.resolve()) and digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    inputs=[];start=time.perf_counter()
    for row in rows:
        paths={};identity={}
        for role in ['content','reference']:
            a=read(row[role]);p=out/'inputs'/f"{row['index']}-{role}.png";Image.fromarray(a).save(p);paths[role]=p
            identity[role]=dict(path=str(p.relative_to(out)),sha256=digest(p),pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        inputs.append((paths,identity))
    report['input_prepare_seconds']=time.perf_counter()-start
    start=time.perf_counter();encoder=Encoder(k_dim=8195,input_dim=4,hidden=1024,output_dim=3,device=torch.device('cuda'))
    state=torch.load(checkpoint,map_location='cuda',weights_only=True);encoder.load_state_dict(state,strict=True);encoder.eval().requires_grad_(False)
    assert all(torch.isfinite(v).all() for v in state.values());report['state_tensors']=len(state);report['parameters']=sum(p.numel() for p in encoder.parameters())
    torch.cuda.synchronize();report['model_load_seconds']=time.perf_counter()-start;torch.cuda.reset_peak_memory_stats()
    predictions={}
    for i,(row,(paths,identity)) in enumerate(zip(rows,inputs)):
        torch.cuda.synchronize();start=time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):imgs=run_inference(encoder,torch.device('cuda'),paths['content'],paths['reference'],compress=False,enc_steps=8,strength=1.,crop=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;pixels=np.array(imgs[2]);assert pixels.shape==(256,256,3) and pixels.dtype==np.uint8
        if i==0:
            with Image.open(paths['content']) as im:
                a=np.array(im.resize((528,528),Image.Resampling.BILINEAR),dtype=np.float32)/255
                # Check the released reshape convention instead of silently replacing it.
                assert np.array_equal(enc_preprocess(im).numpy(),a.reshape(3,528,528))
            with contextlib.redirect_stdout(io.StringIO()):again=run_inference(encoder,torch.device('cuda'),paths['content'],paths['reference'],compress=False,enc_steps=8,strength=1.,crop=False)
            assert np.array_equal(pixels,np.array(again[2]));report['first_exact_repeat']=True
        path=out/'candidate'/f"{row['index']}.png";Image.fromarray(pixels).save(path);predictions[row['index']]=pixels
        report['rows'].append(dict(index=row['index'],inputs=identity,sha256=digest(path),seconds=seconds))
        (out/'progress.json').write_text(json.dumps(report,indent=2))
        if (i+1)%8==0:print('predictions',i+1,flush=True)
    assert all(torch.equal(v,state[k]) for k,v in encoder.state_dict().items());report['model_state_unchanged']=True
    report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();truths={r['index']:read(r['gt']) for r in rows};baseline={}
    for e in baseline_record['cases']:
        path=old/f"{e['index']}.png";assert digest(path)==e['sha256']
        with Image.open(path) as im:baseline[e['index']]=np.array(im)
    scores=score_outputs(rows,predictions,truths);baseline_scores=score_outputs(rows,baseline,truths)
    def no_hashes(s):return dict(s,cases=[{k:v for k,v in r.items() if k!='sha256'} for r in s['cases']])
    assert baseline_scores==no_hashes(baseline_record)
    for e in scores['cases']:e['sha256']=digest(out/'candidate'/f"{e['index']}.png")
    report.update(scores=scores,baseline=baseline_scores,gate=response_development_gate(baseline_scores,scores,baseline_scores))
    verified=0
    for row in report['rows']:
        for e in row['inputs'].values():
            p=out/e['path'];assert digest(p)==e['sha256']
            with Image.open(p) as im:assert hashlib.sha256(np.array(im).tobytes()).hexdigest()==e['pixel_sha256']
        p=out/'candidate'/f"{row['index']}.png";assert digest(p)==row['sha256']
        with Image.open(p) as im:assert np.array_equal(np.array(im),predictions[row['index']]);verified+=1
    for n,h in report['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    report.update(finished=True,prediction_pngs_verified=verified)
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-modflows-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(finished=True,mean=scores['mean'],baseline_mean=baseline_scores['mean'],gate=report['gate'])),flush=True)


if __name__=='__main__':main()
