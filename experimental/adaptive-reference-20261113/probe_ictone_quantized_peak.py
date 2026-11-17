"""Bound largest published compressed linear shapes with synthetic weights."""
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
os.environ['DIFFUSERS_GGUF_CUDA_KERNELS'] = 'false'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    import gc
    import torch
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
    torch.manual_seed(42)
    assert torch.cuda.mem_get_info()[0] >= 4_500_000_000, 'insufficient free VRAM for bounded synthetic probe'
    rows = []
    started = time.monotonic()
    # Largest packed Q3_K weights read from the pinned headers. Fill triptych
    # at 1536x512 gives 3072 image tokens; T5 has 512 text tokens.
    for label, input_width, output_width, tokens in [('t5', 4096, 10240, 512), ('fill_fused', 3072, 21504, 3072)]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        free = torch.cuda.mem_get_info()[0]
        packed = torch.randint(0, 256, (output_width, input_width // 256, 110), device='cuda', dtype=torch.uint8)
        packed[:, :, -2:] = torch.tensor([.0001], dtype=torch.float16, device='cuda').view(torch.uint8)
        layer = GGUFLinear(input_width, output_width, compute_dtype=torch.bfloat16, device='cuda')
        layer.weight = GGUFParameter(packed.reshape(output_width, -1), quant_type=gguf.GGMLQuantizationType.Q3_K)
        inputs = torch.randn(1, tokens, input_width, device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        then = time.monotonic()
        with torch.inference_mode():
            result = layer(inputs)
        torch.cuda.synchronize()
        assert torch.isfinite(result).all()
        rows.append(dict(label=label, logical_weight_shape=[output_width,input_width], input_tokens=tokens,
                         source='synthetic Q3_K bytes using header maximum shape', output_shape=list(result.shape),
                         packed_bytes=packed.numel(), before_allocated_bytes=before, driver_free_before_bytes=free,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(), linear_seconds=time.monotonic()-then,
                         finite=True))
        del result, layer, inputs, packed
    elapsed = time.monotonic() - started
    assert elapsed <= 60, elapsed
    result = dict(finished=True, scope='synthetic largest fused weight only; not full blocks/pipeline or LoRA activation peak',
                  rows=rows, wall_compute_seconds=elapsed,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
