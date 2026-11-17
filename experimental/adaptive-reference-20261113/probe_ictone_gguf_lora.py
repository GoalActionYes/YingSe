"""Small real Q3_K/Diffusers/PEFT execution, not an ICTone quality test."""
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
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import gguf
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.loaders import PeftAdapterMixin
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

    class Tiny(ModelMixin, PeftAdapterMixin):
        def __init__(self, packed):
            super().__init__()
            self.proj = GGUFLinear(256, 6, compute_dtype=torch.bfloat16)
            self.proj.weight = GGUFParameter(packed, quant_type=gguf.GGMLQuantizationType.Q3_K)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(42)
    packed = torch.randint(0, 256, (6, 110), dtype=torch.uint8)
    packed[:, -2:] = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
    base_hash = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
    rows = []
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        model = Tiny(packed.clone()).to(device)
        x = torch.randn(2, 3, 256, device=device, dtype=torch.bfloat16)
        expected_weight = torch.from_numpy(gguf.dequantize(packed.numpy(), gguf.GGMLQuantizationType.Q3_K)).to(device)
        actual_weight = dequantize_gguf_tensor(model.proj.weight)
        # Diffusers dequantizes Q3_K in FP16; gguf's NumPy path uses FP32.
        # Bound FP16 rounding separately from the exact LoRA/disable checks.
        torch.testing.assert_close(actual_weight.float(), expected_weight, rtol=.002, atol=.001)
        before = model(x)
        state = {'proj.lora_A.weight': torch.randn(4, 256, device=device, dtype=torch.bfloat16) * .02,
                 'proj.lora_B.weight': torch.randn(6, 4, device=device, dtype=torch.bfloat16) * .02}
        model.load_lora_adapter(state, prefix=None, adapter_name='ictone', low_cpu_mem_usage=True)
        layer = model.proj
        after = model(x)
        a, b = layer.lora_A['ictone'].weight, layer.lora_B['ictone'].weight
        explicit = before + F.linear(F.linear(x.to(a.dtype), a), b) * layer.scaling['ictone']
        explicit = explicit.to(before.dtype)
        torch.testing.assert_close(after, explicit, rtol=0, atol=0)
        assert (after - before).abs().max().item() > 0
        model.disable_adapters()
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        assert isinstance(layer.base_layer, GGUFLinear)
        assert layer.base_layer.weight.dtype == torch.uint8
        current_hash = hashlib.sha256(layer.base_layer.weight.as_tensor().cpu().numpy().tobytes()).hexdigest()
        assert current_hash == base_hash
        rows.append(dict(device=device, quant_type='Q3_K', adapter_dtypes=[str(a.dtype), str(b.dtype)],
                         dequant_numpy_max_error=float((actual_weight.float()-expected_weight).abs().max()),
                         scaling=layer.scaling['ictone'], nonzero_output_max=float((after-before).abs().max()),
                         explicit_max_error=float((after-explicit).abs().max()), disabled_exact=True,
                         base_bytes_unchanged=True, packed_shape=list(packed.shape), logical_shape=[6, 256]))
    elapsed = time.monotonic() - started
    assert elapsed <= 60, elapsed
    result = dict(finished=True, scope='synthetic linear layer only; no model/image quality claim',
                  rows=rows, wall_seconds=elapsed,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
