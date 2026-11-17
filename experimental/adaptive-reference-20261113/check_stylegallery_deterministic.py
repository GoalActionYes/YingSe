"""Strict deterministic full-size native loss comparison; no candidate output."""
import os,sys,json,time,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ASSETS=ROOT/'.tools/StyleGallery-audit';BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(ASSETS/'mpl-cache'))
sys.path[:0]=[str(ASSETS/'vendor'),str(ROOT/'.tools/ModFlows-audit/vendor'),str(ASSETS/'source')]
import torch
from diffusers import UNet2DConditionModel,DDIMScheduler
from transformers import CLIPTextModel,CLIPTokenizer
import pipeline,utils
from screen_stylegallery import digest


def main():
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=ASSETS/'source/pretrained_models/runwayml_stable-diffusion-v1-5'
    kwargs=dict(torch_dtype=torch.float16,device_map={'':'cuda'},low_cpu_mem_usage=True,local_files_only=True,use_safetensors=True)
    model=UNet2DConditionModel.from_pretrained(str(folder/'unet'),**kwargs).eval().requires_grad_(False)
    text=CLIPTextModel.from_pretrained(str(folder/'text_encoder'),**kwargs).eval().requires_grad_(False);tok=CLIPTokenizer.from_pretrained(str(folder/'tokenizer'),local_files_only=True)
    with torch.no_grad():embeds=text(tok('',padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.cuda())[0]
    del text;gc.collect();torch.cuda.empty_cache()
    # Trusted self-generated caches from our own stopped run; no external pickle.
    caches=[torch.load(BASE/'stylegallery-screen-v3/processed_data/inputs'/f'22927-{role}.pt',map_location='cpu',weights_only=False) for role in ['content','reference']]
    content,style=[r['feature'].cuda().half() for r in caches];initial=caches[0]['C15'].cuda();initial=(initial-initial.mean(1,keepdim=True))/(initial.std(1,keepdim=True)+1e-8)
    masks=[r['mask'].cuda() for r in caches];labels=[torch.unique(m).tolist() for m in masks]
    pairs={c:dict(style_dict_index=0,style_cluster=labels[1][i%len(labels[1])]) for i,c in enumerate(labels[0])}
    cm={c:(masks[0]==c).float() for c in labels[0]};sm={(0,s):(masks[1]==s).float() for s in labels[1]}
    scheduler=DDIMScheduler.from_pretrained(str(folder/'scheduler'),local_files_only=True);scheduler.set_timesteps(150);t=scheduler.timesteps[0]
    trials=[];gradients=[]
    for label,enabled in [('default_A',False),('default_B',False),('checkpoint_C',True)]:
        if enabled:model.enable_gradient_checkpointing()
        else:model.disable_gradient_checkpointing()
        controller=utils.Controller(self_layers=(10,16));cache=utils.AttnDataCache();utils.register_attn_control(model,controller,cache)
        def features(x):
            cache.clear();controller.step()
            with torch.autocast('cuda',dtype=torch.float16):model(x,t,embeds)
            return cache.get()
        with torch.no_grad():qc,_,_,_=features(content);_,ks,vs,_=features(style)
        x=initial.detach().clone().requires_grad_();gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        q,k,v,out=features(x);loss=utils.content_loss(q,qc)*.26+pipeline.StyleGallery.get_style_loss(None,pairs,cm,sm,q,out,[dict(k=ks,v=vs)],{})
        assert torch.isfinite(loss);scale=torch.tensor(1.,device='cuda');(loss*scale).backward();torch.cuda.synchronize();gradient=x.grad.detach()/scale;assert torch.isfinite(gradient).all()
        gradients.append(gradient.cpu().clone());trials.append(dict(label=label,checkpoint_enabled=enabled,loss=float(loss.detach()),incremental_peak_bytes=torch.cuda.max_memory_allocated()-base,seconds=time.perf_counter()-start,gradient_sha256=hashlib.sha256(gradients[-1].numpy().tobytes()).hexdigest()))
        del q,k,v,out,loss,x,qc,ks,vs,gradient;cache.clear();gc.collect();torch.cuda.empty_cache()
    comparisons={}
    for name,i,j in [('default_repeat',0,1),('checkpoint_vs_default',0,2)]:
        d=(gradients[i]-gradients[j]).abs();comparisons[name]=dict(loss_exact=trials[i]['loss']==trials[j]['loss'],gradient_exact=torch.equal(gradients[i],gradients[j]),maximum_gradient_difference=float(d.max()),mean_gradient_difference=float(d.mean()))
    report=dict(scope='Full64x64 latent single-step native loss only; constructed fixed region matching, fixed scale1, no candidate output',fixed_gradient_scale=1,runtime=torch.__version__,trials=trials,comparisons=comparisons,source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['check_stylegallery_deterministic.py','stylegallery-deterministic-protocol.md']},training=False,product_integrated=False)
    report.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],passed=all(c['loss_exact'] and c['gradient_exact'] for c in comparisons.values()))
    (ROOT/'docs/research/2026-11-17-stylegallery-deterministic-evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True);assert report['passed']


if __name__=='__main__':main()
