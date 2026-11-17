"""Real frozen UNet/cache gradient and memory comparison, synthetic latent only."""
import os,sys,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ASSETS=ROOT/'.tools/StyleGallery-audit'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(ASSETS/'mpl-cache'))
sys.path[:0]=[str(ASSETS/'vendor'),str(ROOT/'.tools/ModFlows-audit/vendor'),str(ASSETS/'source')]
import torch
from diffusers import UNet2DConditionModel
import utils
from screen_stylegallery import digest


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=UNet2DConditionModel.from_pretrained(str(ASSETS/'source/pretrained_models/runwayml_stable-diffusion-v1-5/unet'),torch_dtype=torch.float16,device_map={'':'cuda'},low_cpu_mem_usage=True,local_files_only=True,use_safetensors=True).eval().requires_grad_(False)
    # State clones stay on CPU one tensor at a time only as hashes, not a second full model.
    def identity():
        import hashlib
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=identity();torch.manual_seed(20261117);latent=torch.randn((1,4,16,16),device='cuda');reference=torch.randn_like(latent);embeds=torch.randn((1,77,768),device='cuda',dtype=torch.float16)
    trials=[];outputs=[]
    for enabled in [False,True]:
        if enabled:model.enable_gradient_checkpointing()
        else:model.disable_gradient_checkpointing()
        controller=utils.Controller(self_layers=(10,16));cache=utils.AttnDataCache();utils.register_attn_control(model,controller,cache)
        def features(x):
            cache.clear();controller.step()
            with torch.autocast('cuda',dtype=torch.float16):model(x,500,embeds)
            return cache.get()
        with torch.no_grad():qc,_,_,_=features(latent);_,ks,vs,_=features(reference)
        x=latent.detach().clone().requires_grad_();gc.collect();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        q,k,v,out=features(x);shapes=[list(t.shape) for t in q];loss=utils.content_loss(q,qc)*.26+utils.style_loss(q,ks,vs,out)
        assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();assert torch.isfinite(x.grad).all()
        outputs.append((loss.detach().cpu().clone(),x.grad.detach().cpu().clone()))
        trials.append(dict(checkpoint_enabled=enabled,loss=float(loss.detach()),q_shapes=shapes,incremental_peak_bytes=torch.cuda.max_memory_allocated()-base,seconds=time.perf_counter()-start))
        del q,k,v,out,loss,x,qc,ks,vs;cache.clear();gc.collect();torch.cuda.empty_cache()
    result=dict(scope='Synthetic16x16 latent real UNet/native attention-cache execution test only',trials=trials,loss_exact=torch.equal(outputs[0][0],outputs[1][0]),gradient_exact=torch.equal(outputs[0][1],outputs[1][1]),maximum_gradient_difference=float((outputs[0][1]-outputs[1][1]).abs().max()),cache_shapes_exact=trials[0]['q_shapes']==trials[1]['q_shapes'],model_state_unchanged=before==identity(),incremental_peak_reduced=trials[1]['incremental_peak_bytes']<trials[0]['incremental_peak_bytes'],source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['check_stylegallery_checkpoint.py','stylegallery-checkpoint-protocol.md']},training=False,product_integrated=False)
    result['passed']=all(result[k] for k in ['loss_exact','gradient_exact','cache_shapes_exact','model_state_unchanged','incremental_peak_reduced'])
    (ROOT/'docs/research/2026-11-17-stylegallery-checkpoint-evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    assert result['passed']


if __name__=='__main__':main()
