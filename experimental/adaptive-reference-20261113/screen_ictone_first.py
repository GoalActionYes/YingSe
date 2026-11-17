"""First real offline ICTone Q3 image, frozen README config and no GT access."""
import argparse
import ast
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from ictone_flux_stage import ROOT, ASSETS, stage_real_flux, reload_staged_flux


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for body in iter(lambda: handle.read(1_048_576), b''):
            h.update(body)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--index',type=int,default=0,help='Ordinal within the fixed first eight development rows')
    p.add_argument('--reuse-stage',help='Previously audited first-case directory; never rewrite its groups')
    p.add_argument('--input-spec',help='Strict RGB-only input JSON for a separately frozen protocol')
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113/tst100k' / args.run
    out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    import psutil
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import FluxFillPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.quantizers.gguf.utils import GGUFParameter
    from screen_stylegallery import checkpoint_key_map
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    status = dict(finished=False, scope='offline research; first consumed development image; not product candidate',
                  config=dict(steps=4, guidance=50., seed=666, panel=[512,512], source='author pinned README'),
                  source_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                 ['screen_ictone_first.py','ictone_flux_stage.py','ictone_gguf_runtime.py','ictone_input_spec.py']})
    for name in status['source_sha256']:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def stage(name):
        status.update(stage=name, elapsed_seconds=time.monotonic()-started)
        (out / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(name, flush=True)
    try:
        base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
        if args.input_spec:
            from ictone_input_spec import read_rgb_input_spec
            index,canonical=read_rgb_input_spec(args.input_spec,base)
            status.update(index=index,input_spec_sha256=digest(args.input_spec))
        else:
            data = base / 'cohort-training-coverage-v2'
            assert digest(data/'cohort.json') == 'ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
            assert digest(data/'srgb-v1/manifest.json') == 'da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
            rows = json.loads((data/'cohort.json').read_text())['groups']['validation']
            assert 0<=args.index<8
            row = rows[args.index]
            status['index']=row['index']
            lookup = {x['original_path']:x for x in json.loads((data/'srgb-v1/manifest.json').read_text())['files']}
            canonical={}
            for role in ['content','reference']:
                item=lookup[row[role]]
                path=data/item['prepared_path']
                assert path.resolve().is_relative_to(base.resolve()) and digest(path)==item['sha256']
                with Image.open(path) as im:
                    canonical[role]=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        inputs = {}
        for role in ['content','reference']:
            pixels=canonical[role]
            pixels.save(out/(role+'256.png'))
            inputs[role] = pixels.resize((512,512), Image.Resampling.LANCZOS)
        status['inputs'] = {role:dict(sha256=digest(out/(role+'256.png'))) for role in inputs}
        stage('staging_real_fill_lora')
        if args.reuse_stage:
            assert Path(args.reuse_stage).name==args.reuse_stage
            source=base/args.reuse_stage
            load_start=time.monotonic()
            model,staged=reload_staged_flux(source)
            (out/'stage.json').write_bytes((source/'stage.json').read_bytes())
            status.update(reused_stage_from=args.reuse_stage,staging_seconds=0.,reload_seconds=time.monotonic()-load_start)
        else:
            model, staged = stage_real_flux(out)
            status['staging_seconds'] = staged['wall_seconds']
        components = ASSETS/'models/components'
        stage('loading_clip_and_vae')
        clip = CLIPTextModel.from_pretrained(components/'text_encoder', torch_dtype=torch.bfloat16,
                    device_map={'':'cuda'}, low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True).eval()
        vae = AutoencoderKL.from_pretrained(components/'vae', torch_dtype=torch.bfloat16,
                    device_map={'':'cuda'}, low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True).eval()
        verified = {}
        for label, component, filename in [('clip',clip,'text_encoder/model.safetensors'),('vae',vae,'vae/diffusion_pytorch_model.safetensors')]:
            with safe_open(str(components/filename), framework='pt', device='cuda') as source:
                mapping = checkpoint_key_map(component, source.keys())
                state = component.state_dict()
                assert set(mapping) == set(state), (label,set(mapping)^set(state))
                for name, actual in state.items():
                    expected = source.get_tensor(mapping[name]).to(actual.dtype)
                    assert torch.equal(actual,expected) and torch.isfinite(actual).all(),(label,name)
                verified[label] = len(state)
        status['component_state_tensors_verified'] = verified
        cache_folder = ROOT/'outputs/adaptive-reference-20261113/ictone-real-t5-v1'
        cache_report = json.loads((cache_folder/'results.json').read_text())
        assert digest(cache_folder/'prompt.safetensors') == cache_report['cache_sha256']
        assert json.loads((cache_folder/'audit.json').read_text())['result_sha256'] == digest(cache_folder/'results.json')
        prompt = cache_report['prompt']
        prompt_embeds = load_file(str(cache_folder/'prompt.safetensors'),device='cuda')['prompt_embeds']
        tokenizer = CLIPTokenizer.from_pretrained(components/'tokenizer',local_files_only=True)
        ids = tokenizer(prompt,padding='max_length',max_length=77,truncation=True,return_tensors='pt').input_ids.to('cuda')
        with torch.inference_mode():
            pooled = clip(ids,output_hidden_states=False).pooler_output
        assert pooled.shape == (1,768) and torch.isfinite(pooled).all()
        del clip
        gc.collect()
        torch.cuda.empty_cache()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(components/'scheduler',local_files_only=True)
        pipe = FluxFillPipeline(scheduler=scheduler,vae=vae,text_encoder=None,tokenizer=tokenizer,
                               text_encoder_2=None,tokenizer_2=None,transformer=model)
        author = ROOT/'.tools/ICTone-audit/inference.py'
        assert digest(author) == '0899f20276cfe6e3da42426fc511a96644440b49c721824d188e37760ee1bcca'
        tree = ast.parse(author.read_bytes())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_triptych')
        namespace = {'Image':Image,'np':np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(author),'exec'),namespace)
        tri,mask = namespace['build_triptych'](inputs['content'],inputs['reference'],512,512)
        tri.save(out/'triptych-input.png')
        mask.save(out/'mask.png')
        expected_hash = {x['name']:x['sha256'] for group in staged['groups'] for x in group['state']}
        checked = set()
        hooks = []
        then = time.monotonic()
        def make_check(name):
            def check(module,args):
                assert time.monotonic()-then <= 600, '600-second image generation budget exhausted'
                for local_name,parameter in module.named_parameters(recurse=False):
                    full = name+'.'+local_name
                    if full in checked:
                        continue
                    value = parameter.as_tensor() if isinstance(parameter,GGUFParameter) else parameter
                    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    assert hashlib.sha256(raw).hexdigest() == expected_hash[full],full
                    checked.add(full)
            return check
        for name, layer in model.named_modules():
            if list(layer.parameters(recurse=False)):
                hooks.append(layer.register_forward_pre_hook(make_check(name)))
        status['initial_resources'] = dict(ram_available=psutil.virtual_memory().available,cuda_free_total=list(torch.cuda.mem_get_info()))
        stage('generating_first_image')
        torch.cuda.reset_peak_memory_stats()
        def callback(pipe,step,timestep,kwargs):
            assert torch.isfinite(kwargs['latents']).all()
            status['completed_steps'] = step+1
            stage('generation_step_'+str(step+1))
            return kwargs
        with torch.inference_mode():
            result = pipe(prompt_embeds=prompt_embeds,pooled_prompt_embeds=pooled,image=tri,mask_image=mask,
                          width=1536,height=512,guidance_scale=50.,num_inference_steps=4,max_sequence_length=512,
                          generator=torch.Generator(device='cpu').manual_seed(666),
                          callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents']).images[0]
        torch.cuda.synchronize()
        status['image_pipeline_seconds'] = time.monotonic()-then
        for hook in hooks:
            hook.remove()
        assert len(checked)==staged['parameter_count']==1846,(len(checked),staged['parameter_count'])
        assert result.size==(1536,512)
        result.save(out/'triptych-output.png')
        native = result.crop((1024,0,1536,512))
        native.save(out/'native512.png')
        native.resize((256,256),Image.Resampling.BILINEAR).save(out/'metric256.png')
        status.update(finished=True,checked_parameter_count=len(checked),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      output_sha256={n:digest(out/n) for n in ['triptych-output.png','native512.png','metric256.png']})
        stage('finished')
        (out/'results.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    except Exception as error:
        status.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        stage('failed')
        raise


if __name__=='__main__':
    main()
