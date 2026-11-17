"""Measure allocator history and exact native single-step values; no candidate image."""
import os,sys,json,time,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ASSETS=ROOT/'.tools/StyleGallery-audit';BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(ASSETS/'mpl-cache'))
sys.path[:0]=[str(ASSETS/'vendor'),str(ROOT/'.tools/ModFlows-audit/vendor'),str(ASSETS/'source')]
import torch,numpy as np,psutil
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
from transformers import CLIPTextModel,CLIPTokenizer,CLIPImageProcessor
from accelerate import Accelerator
import pipeline,utils
from screen_stylegallery import digest


def main():
    torch.use_deterministic_algorithms(True,warn_only=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=BASE/'stylegallery-allocator-v1';out.mkdir(exist_ok=False);started=time.perf_counter();report=dict(scope='FP32-to-FP16 partial preparation/allocator probe; no candidate output',snapshots=[],trials=[],finished=False)
    def snapshot(name):
        torch.cuda.synchronize();s=dict(stage=name,seconds=time.perf_counter()-started,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),cuda_free_total=list(torch.cuda.mem_get_info()),ram_available=psutil.virtual_memory().available)
        report['snapshots'].append(s);(out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(s),flush=True);assert s['seconds']<180
        return s
    folder=ASSETS/'source/pretrained_models/runwayml_stable-diffusion-v1-5';kwargs=dict(torch_dtype=torch.float32,device_map={'':'cuda'},low_cpu_mem_usage=True,local_files_only=True,use_safetensors=True)
    text=CLIPTextModel.from_pretrained(str(folder/'text_encoder'),**kwargs);unet=UNet2DConditionModel.from_pretrained(str(folder/'unet'),**kwargs);vae=AutoencoderKL.from_pretrained(str(folder/'vae'),**kwargs)
    scheduler=DDIMScheduler.from_pretrained(str(folder/'scheduler'),local_files_only=True);tokenizer=CLIPTokenizer.from_pretrained(str(folder/'tokenizer'),local_files_only=True);processor=CLIPImageProcessor.from_pretrained(str(folder/'feature_extractor'),local_files_only=True)
    pipe=pipeline.StyleGallery(vae=vae,text_encoder=text,tokenizer=tokenizer,unet=unet,scheduler=scheduler,safety_checker=None,feature_extractor=processor,requires_safety_checker=False);pipe.extractor=pipe.unet;pipe.freeze();snapshot('loaded_fp32')
    paths=[BASE/'stylegallery-screen-deterministic-baseline-v1/processed_data/inputs'/f'22927-{r}.pt' for r in ['content','reference']]
    # Trusted self-generated local caches, not downloaded pickle.
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];report['cache_sha256']=[digest(p) for p in paths]
    c,s=[r['feature'].cuda() for r in records];scheduler.set_timesteps(150);t=scheduler.timesteps[0]
    with torch.no_grad():
        embeds32=pipe.encode_prompt('',pipe.device,1,False)[0];dummy=unet(c,t,embeds32)[0];del dummy
        dummy=vae.decode(c/vae.config.scaling_factor)[0];del dummy,embeds32
    snapshot('partial_fp32_preparation_complete')
    pipe.accelerator=Accelerator(mixed_precision='fp16',gradient_accumulation_steps=1);pipe.init(False);snapshot('converted_fp16')
    controller=utils.Controller(self_layers=(10,16));cache=utils.AttnDataCache();utils.register_attn_control(pipe.extractor,controller,cache);pipe.extractor=pipe.accelerator.prepare(pipe.extractor)
    with torch.no_grad():embeds=pipe.encode_prompt('',pipe.device,1,False)[0]
    def identity():
        h=hashlib.sha256()
        for name,module in [('unet',pipe.unet),('vae',pipe.vae),('text',pipe.text_encoder)]:
            for key,value in module.state_dict().items():h.update((name+key).encode());h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=identity();initial=records[0]['C15'].cuda();initial=(initial-initial.mean(1,keepdim=True))/(initial.std(1,keepdim=True)+1e-8);c=c.half()
    masks=[r['mask'].cuda() for r in records];labels=[torch.unique(m).tolist() for m in masks];matches={key:dict(style_dict_index=0,style_cluster=labels[1][i%len(labels[1])]) for i,key in enumerate(labels[0])}
    cm={key:(masks[0]==key).float() for key in labels[0]};sm={(0,key):(masks[1]==key).float() for key in labels[1]}
    def features(x):
        cache.clear();controller.step();pipe.extractor(x,t,embeds);return cache.get()
    values=[];snap_before=None;snap_after=None
    for name,trim in [('untrimmed_A',False),('untrimmed_B',False),('trimmed_C',True)]:
        if trim:
            snap_before=snapshot('before_cache_release');gc.collect();torch.cuda.empty_cache();snap_after=snapshot('after_cache_release')
        with torch.no_grad():qc,_,_,_=features(c);_,ks,vs,_=features(s)
        x=initial.detach().clone().requires_grad_();snapshot(name+'_before');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        q,k,v,self_out=features(x);loss=utils.content_loss(q,qc)*.26+pipeline.StyleGallery.get_style_loss(None,matches,cm,sm,q,self_out,[dict(k=ks,v=vs)],{});assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();assert torch.isfinite(x.grad).all()
        values.append((loss.detach().cpu().clone(),x.grad.detach().cpu().clone()));report['trials'].append(dict(name=name,loss=float(loss.detach()),seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),gradient_sha256=hashlib.sha256(values[-1][1].numpy().tobytes()).hexdigest()))
        del q,k,v,self_out,loss,x,qc,ks,vs;cache.clear();snapshot(name+'_after')
    report.update(loss_exact=all(torch.equal(values[0][0],e[0]) for e in values[1:]),gradient_exact=all(torch.equal(values[0][1],e[1]) for e in values[1:]),state_unchanged=before==identity(),reserved_released=snap_before['reserved']-snap_after['reserved'],allocated_unchanged=snap_before['allocated']==snap_after['allocated'],source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['probe_stylegallery_allocator.py','stylegallery-allocator-protocol.md']},finished=True,product_integrated=False)
    report['supports_full_trial']=report['loss_exact'] and report['gradient_exact'] and report['state_unchanged'] and report['allocated_unchanged'] and report['reserved_released']>=256*1024*1024
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-stylegallery-allocator-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,supports_full_trial=report['supports_full_trial'],reserved_released=report['reserved_released'],trials=report['trials'])),flush=True)


if __name__=='__main__':main()
