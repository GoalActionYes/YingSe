"""Load the pinned real T5 GGUF by tensor and encode the actual author prompt."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.tools/ICTone-quantized-audit'
sys.path[:0] = [str(ASSETS / 'vendor'), str(ROOT / '.tools/StyleGallery-audit/vendor'),
                str(ROOT / '.tools/ModFlows-audit/vendor')]
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
    # Only execute after complete asset hashing; prefixes are insufficient.
    verified = json.loads((ASSETS / 'assets-t5-verified.json').read_text())
    assert verified['finished'] and len(verified['records']) == 1
    record = verified['records'][0]
    weight_path = ASSETS / record['target']
    assert weight_path.stat().st_size == record['bytes']
    import ast
    import numpy as np
    import psutil
    import torch
    import gguf
    from accelerate import init_empty_weights
    from transformers import T5Config, T5EncoderModel, T5TokenizerFast
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor, SUPPORTED_GGUF_QUANT_TYPES
    from ictone_gguf_runtime import adapt_t5_gated_gguf, load_t5_tokenizer_from_ascii_cache
    author = ROOT / '.tools/ICTone-audit/inference.py'
    author_bytes = author.read_bytes()
    assert hashlib.sha256(author_bytes).hexdigest() == '0899f20276cfe6e3da42426fc511a96644440b49c721824d188e37760ee1bcca'
    tree = ast.parse(author_bytes)
    prompt = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_INSTANCE_PROMPT' for t in node.targets))
    tokenizer, tokenizer_content_id = load_t5_tokenizer_from_ascii_cache(ASSETS / 'models/components/tokenizer_2')
    ids = tokenizer(prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt').input_ids.to('cuda')
    assert int((ids != tokenizer.pad_token_id).sum()) > 1
    header_report = json.loads((ROOT / 'outputs/adaptive-reference-20261113/ictone-t5-header-v2/results.json').read_text())
    config = T5Config.from_dict(header_report['config'])
    with init_empty_weights():
        model = T5EncoderModel(config).eval().to(dtype=torch.bfloat16)
    mapping = get_gguf_hf_weights_map(model, model_type='t5encoder', num_layers=24)
    reader = gguf.GGUFReader(str(weight_path))
    assert len(reader.tensors) == 219
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    rows = []
    with weight_path.open('rb') as handle:
        for tensor in reader.tensors:
            assert time.monotonic() - started <= 600, 'loading budget exhausted'
            name = mapping[tensor.name]
            parent, _, leaf = name.rpartition('.')
            module = model.get_submodule(parent)
            shape = tuple(reversed(tensor.shape.tolist()))
            handle.seek(tensor.data_offset)
            body = handle.read(tensor.n_bytes)
            assert len(body) == tensor.n_bytes
            array = np.frombuffer(body, dtype=tensor.data.dtype).reshape(tensor.data.shape).copy()
            raw = torch.from_numpy(array).to('cuda')
            quantized = tensor.tensor_type in SUPPORTED_GGUF_QUANT_TYPES
            if quantized:
                parameter = GGUFParameter(raw, quant_type=tensor.tensor_type)
                if isinstance(module, torch.nn.Linear):
                    assert leaf == 'weight' and module.bias is None
                    with init_empty_weights():
                        replacement = GGUFLinear(shape[1], shape[0], compute_dtype=torch.bfloat16)
                    replacement.weight = parameter
                    ancestor, _, module_leaf = parent.rpartition('.')
                    setattr(model.get_submodule(ancestor), module_leaf, replacement)
                else:
                    assert isinstance(module, torch.nn.Embedding), (name, type(module))
                    dense = dequantize_gguf_tensor(parameter).to(torch.bfloat16)
                    assert tuple(dense.shape) == shape
                    setattr(module, leaf, torch.nn.Parameter(dense, requires_grad=False))
                    del dense
                del parameter
            else:
                assert tensor.tensor_type in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
                assert tuple(raw.shape) == shape
                setattr(module, leaf, torch.nn.Parameter(raw.to(torch.bfloat16), requires_grad=False))
            rows.append(dict(gguf_name=tensor.name, hf_name=name, shape=list(shape),
                             quant_type=int(tensor.tensor_type), payload_sha256=hashlib.sha256(body).hexdigest()))
            peak_rss = max(peak_rss, process.memory_info().rss)
            del raw, array, body
    assert all(not p.is_meta and p.device.type == 'cuda' for p in model.parameters())
    assert model.shared is model.encoder.embed_tokens
    assert len(adapt_t5_gated_gguf(model)) == 24
    model.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    load_seconds = time.monotonic() - started
    then = time.monotonic()
    with torch.inference_mode():
        embedding = model(ids, output_hidden_states=False)[0]
    torch.cuda.synchronize()
    encode_seconds = time.monotonic() - then
    assert encode_seconds <= 600 and embedding.shape == (1, 512, 4096)
    assert torch.isfinite(embedding).all()
    # Repeat once to establish stable cached prompt identity, without changing
    # seed, text or decoder settings. This is not image inference.
    with torch.inference_mode():
        again = model(ids, output_hidden_states=False)[0]
    torch.testing.assert_close(again, embedding, rtol=0, atol=0)
    from safetensors.torch import save_file
    save_file({'prompt_embeds': embedding.cpu().contiguous(), 'input_ids': ids.cpu().contiguous()}, str(out / 'prompt.safetensors'))
    result = dict(finished=True, scope='real quantized T5 prompt only, no image or baseline quality claim',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_sha256=hashlib.sha256(Path(__file__).with_name('ictone_gguf_runtime.py').read_bytes()).hexdigest(),
                  weight_sha256=record['sha256'], prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  nonpad_tokens=int((ids != tokenizer.pad_token_id).sum()), loaded=rows,
                  tokenizer_content_id=tokenizer_content_id,
                  load_seconds=load_seconds, first_encode_seconds=encode_seconds, repeat_exact=True,
                  sampled_peak_rss_bytes=peak_rss, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  cache_sha256=hashlib.sha256((out / 'prompt.safetensors').read_bytes()).hexdigest())
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['loaded', 'prompt']}, indent=2))


if __name__ == '__main__':
    main()
