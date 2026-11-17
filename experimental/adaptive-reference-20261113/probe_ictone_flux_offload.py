"""Small full Fill-shaped transformer with real LoRA and Q3_K disk groups."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.tools/ICTone-quantized-audit/vendor'),
                str(ROOT / '.tools/StyleGallery-audit/vendor'), str(ROOT / '.tools/ModFlows-audit/vendor')]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DIFFUSERS_GGUF_CUDA_KERNELS'] = 'false'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    import torch
    import gguf
    from diffusers import FluxTransformer2DModel
    from diffusers.hooks import apply_group_offloading
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
    from ictone_gguf_runtime import release_disk_group_storage
    started = time.monotonic()
    torch.manual_seed(42)
    model = FluxTransformer2DModel(in_channels=384, out_channels=64,
            num_layers=2, num_single_layers=2, attention_head_dim=128,
            num_attention_heads=2, joint_attention_dim=256,
            pooled_projection_dim=128, guidance_embeds=True).eval().to(dtype=torch.bfloat16)
    replaced = []
    adapters = {}
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, torch.nn.Linear):
            continue
        if layer.in_features % 256 == 0:
            packed = torch.randint(0, 256, (layer.out_features, layer.in_features // 256, 110), dtype=torch.uint8)
            packed[:, :, -2:] = torch.tensor([.0001], dtype=torch.float16).view(torch.uint8)
            quant = GGUFLinear(layer.in_features, layer.out_features, bias=layer.bias is not None,
                               compute_dtype=torch.bfloat16)
            quant.weight = GGUFParameter(packed.reshape(layer.out_features, -1), quant_type=gguf.GGMLQuantizationType.Q3_K)
            if layer.bias is not None:
                quant.bias = torch.nn.Parameter(layer.bias.detach().clone())
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, quant)
            replaced.append(name)
        # Match the actual adapter's use of attention, FFN, norm linear and
        # x_embedder modules, including unquantized Fill input projection.
        if name.startswith(('transformer_blocks.', 'single_transformer_blocks.')) or name == 'x_embedder':
            adapters[name + '.lora_A.weight'] = torch.randn(8, layer.in_features, dtype=torch.bfloat16) * .02
            adapters[name + '.lora_B.weight'] = torch.randn(layer.out_features, 8, dtype=torch.bfloat16) * .02
    model.load_lora_adapter(adapters, prefix=None, adapter_name='ictone', low_cpu_mem_usage=True)
    model.requires_grad_(False).to('cuda')
    inputs = dict(hidden_states=torch.randn(1, 6, 384, device='cuda', dtype=torch.bfloat16),
                  encoder_hidden_states=torch.randn(1, 4, 256, device='cuda', dtype=torch.bfloat16),
                  pooled_projections=torch.randn(1, 128, device='cuda', dtype=torch.bfloat16),
                  timestep=torch.tensor([.5], device='cuda', dtype=torch.bfloat16),
                  guidance=torch.ones(1, device='cuda', dtype=torch.bfloat16),
                  img_ids=torch.zeros(6, 3, device='cuda'), txt_ids=torch.zeros(4, 3, device='cuda'),
                  return_dict=False)
    def hashes():
        return {name: hashlib.sha256((p.as_tensor() if isinstance(p, GGUFParameter) else p).detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, p in model.named_parameters()}
    identity = hashes()
    with torch.inference_mode():
        expected = model(**inputs)[0]
        model.disable_adapters()
        no_lora = model(**inputs)[0]
        model.enable_adapters()
    assert torch.isfinite(expected).all() and not torch.equal(expected, no_lora)
    # PEFT enable_adapters() marks adapter parameters trainable again.
    # Freeze after the enabled/disabled control before immutable disk groups.
    model.requires_grad_(False)
    apply_group_offloading(model, onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                           offload_type='block_level', num_blocks_per_group=1, use_stream=False,
                           offload_to_disk_path=str(out / 'disk-groups'))
    groups = release_disk_group_storage(model)
    restored_names = []
    hooks = []
    # Inspect each module when it is actually onloaded, before computation.
    for name, layer in model.named_modules():
        if not list(layer.parameters(recurse=False)):
            continue
        def check(module, args, name=name):
            for local_name, parameter in module.named_parameters(recurse=False):
                full_name = name + '.' + local_name
                t = parameter.as_tensor() if isinstance(parameter, GGUFParameter) else parameter
                digest = hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert digest == identity[full_name], full_name
                restored_names.append(full_name)
        hooks.append(layer.register_forward_pre_hook(check))
    rows = []
    for iteration in range(2):
        with torch.inference_mode():
            got = model(**inputs)[0]
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert all(p.untyped_storage().nbytes() == 0 for group in groups for p in group.tensor_to_key)
        rows.append(dict(iteration=iteration, output_exact=True, all_group_cpu_storage_bytes=0))
    for hook in hooks:
        hook.remove()
    # Parameters of normalization layers may be used functionally instead of
    # calling their own module. All used modules are checked; group files hold
    # all state, including those parameters, so report coverage explicitly.
    covered = sorted(set(restored_names))
    elapsed = time.monotonic() - started
    assert elapsed <= 60, elapsed
    result = dict(finished=True, scope='synthetic four-block Fill-shaped model, not pretrained pipeline/image quality',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_sha256=hashlib.sha256(Path(__file__).with_name('ictone_gguf_runtime.py').read_bytes()).hexdigest(),
                  q3_linear_count=len(replaced), adapter_tensor_count=len(adapters), disk_groups=len(groups),
                  lora_output_nonzero_max=float((expected-no_lora).abs().max()),
                  checked_onload_parameters=covered, total_parameters=len(identity),
                  rows=rows, wall_compute_seconds=elapsed)
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checked_onload_parameters'}, indent=2))


if __name__ == '__main__':
    main()
