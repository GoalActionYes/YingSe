"""Full small T5 Q3_K execution and stock disk group-offload investigation."""
import argparse
import copy
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--release-cpu-storage', action='store_true')
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    import gguf
    import torch
    from transformers import T5Config, T5EncoderModel
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
    from diffusers.hooks import apply_group_offloading
    from ictone_gguf_runtime import adapt_t5_gated_gguf, release_disk_group_storage
    started = time.monotonic()
    torch.manual_seed(42)
    config = T5Config(vocab_size=32, d_model=256, d_kv=64, d_ff=512,
                      num_layers=2, num_heads=4, feed_forward_proj='gated-gelu',
                      dropout_rate=0, is_encoder_decoder=False, use_cache=False)
    model = T5EncoderModel(config).eval().to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    dense = copy.deepcopy(model)
    replaced = []
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, torch.nn.Linear):
            continue
        assert layer.in_features % 256 == 0 and layer.bias is None
        packed = torch.randint(0, 256, (layer.out_features, layer.in_features // 256, 110), dtype=torch.uint8)
        packed[:, :, -2:] = torch.tensor([.0001], dtype=torch.float16).view(torch.uint8)
        quant = GGUFLinear(layer.in_features, layer.out_features, compute_dtype=torch.bfloat16)
        quant.weight = GGUFParameter(packed.reshape(layer.out_features, -1), quant_type=gguf.GGMLQuantizationType.Q3_K)
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent), leaf, quant)
        replaced.append(name)
    assert len(replaced) == 14
    adapted = adapt_t5_gated_gguf(model)
    assert len(adapted) == 2
    for name in replaced:
        quant = model.get_submodule(name)
        dense.get_submodule(name).weight.data = dequantize_gguf_tensor(quant.weight).to(torch.bfloat16)
    device = torch.device('cuda')
    model.to(device)
    dense.to(device)
    inputs = torch.tensor([[1, 5, 6, 9, 3, 11, 18, 2]], device=device)

    def packed_hashes(module):
        return {name: hashlib.sha256(module.get_submodule(name).weight.as_tensor().cpu().numpy().tobytes()).hexdigest()
                for name in replaced}

    identities = packed_hashes(model)
    with torch.inference_mode():
        expected = dense(inputs).last_hidden_state
        actual = model(inputs).last_hidden_state
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del dense
    torch.cuda.empty_cache()
    apply_group_offloading(model.encoder, onload_device=device, offload_device=torch.device('cpu'),
                           offload_type='block_level', num_blocks_per_group=1, use_stream=False,
                           offload_to_disk_path=str(out / 'disk-groups'))
    released = release_disk_group_storage(model.encoder) if args.release_cpu_storage else []
    observations = []
    for iteration in range(2):
        with torch.inference_mode():
            got = model(inputs).last_hidden_state
        torch.testing.assert_close(got, actual, rtol=0, atol=0)
        quant_states = []
        for name in replaced:
            weight = model.get_submodule(name).weight
            assert isinstance(weight, GGUFParameter)
            assert weight.quant_type == gguf.GGMLQuantizationType.Q3_K
            quant_states.append(dict(name=name, dtype=str(weight.dtype), device=str(weight.device),
                                     storage_bytes=weight.untyped_storage().nbytes(),
                                     logical_bytes=weight.numel() * weight.element_size()))
        if args.release_cpu_storage:
            assert all(p.untyped_storage().nbytes() == 0 for group in released for p in group.tensor_to_key)
        observations.append(dict(iteration=iteration, exact_output=True, quant_states=quant_states,
                                 packed_bytes_equal_after_offload=packed_hashes(model) == identities))
    # Stock disk offload intentionally replaces CPU values with uninitialized
    # full-size storage. Inspect onload too: bytes must be restored before use.
    restored = []
    hooks = []
    for name in replaced:
        def check(module, args, name=name):
            assert module.weight.device.type == 'cuda'
            digest = hashlib.sha256(module.weight.as_tensor().cpu().numpy().tobytes()).hexdigest()
            assert digest == identities[name]
            restored.append(name)
        hooks.append(model.get_submodule(name).register_forward_pre_hook(check))
    with torch.inference_mode():
        third = model(inputs).last_hidden_state
    torch.testing.assert_close(third, actual, rtol=0, atol=0)
    assert sorted(restored) == sorted(replaced)
    for hook in hooks:
        hook.remove()
    elapsed = time.monotonic() - started
    assert elapsed <= 60, elapsed
    result = dict(finished=True, scope='two-layer synthetic T5, not pretrained XXL or image quality',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  replaced_linear_count=len(replaced), dense_dequantized_output_exact=True,
                  adapted_t5_gated_modules=adapted,
                  release_cpu_storage=args.release_cpu_storage, released_disk_groups=len(released),
                  runtime_sha256=hashlib.sha256((Path(__file__).with_name('ictone_gguf_runtime.py')).read_bytes()).hexdigest(),
                  observations=observations, third_forward_restored_all_packed_bytes=True,
                  wall_compute_seconds=elapsed,
                  disk_files=[dict(name=x.name, bytes=x.stat().st_size, sha256=hashlib.sha256(x.read_bytes()).hexdigest())
                              for x in sorted((out / 'disk-groups').glob('*.safetensors'))])
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=True, replaced=len(replaced), exact=True,
                         stock_cpu_storage_bytes=sum(x['storage_bytes'] for x in observations[-1]['quant_states']),
                         stock_cpu_payload_preserved=observations[-1]['packed_bytes_equal_after_offload'],
                         restored_exact=True, seconds=elapsed), indent=2))


if __name__ == '__main__':
    main()
