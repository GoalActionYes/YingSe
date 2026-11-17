"""Verify real T5 tensor identity and cached prompt; does not rerun GPU."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.tools/ICTone-quantized-audit'
sys.path[:0] = [str(ASSETS / 'vendor'), str(ROOT / '.tools/StyleGallery-audit/vendor'),
                str(ROOT / '.tools/ModFlows-audit/vendor')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    report_path = out / 'results.json'
    report = json.loads(report_path.read_text())
    assert report['finished']
    for name, key in [('encode_ictone_t5.py', 'source_sha256'), ('ictone_gguf_runtime.py', 'runtime_sha256')]:
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == report[key]
    import gguf
    import torch
    from safetensors.torch import load_file
    from ictone_gguf_runtime import load_t5_tokenizer_from_ascii_cache
    verified = json.loads((ASSETS / 'assets-t5-verified.json').read_text())['records'][0]
    assert verified['sha256'] == report['weight_sha256']
    path = ASSETS / verified['target']
    reader = gguf.GGUFReader(str(path))
    tensors = {x.name: x for x in reader.tensors}
    assert len(tensors) == len(report['loaded']) == 219
    with path.open('rb') as handle:
        for row in report['loaded']:
            item = tensors[row['gguf_name']]
            assert row['shape'] == list(reversed(item.shape.tolist()))
            assert row['quant_type'] == int(item.tensor_type)
            handle.seek(item.data_offset)
            body = handle.read(item.n_bytes)
            assert hashlib.sha256(body).hexdigest() == row['payload_sha256']
    cache_path = out / 'prompt.safetensors'
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == report['cache_sha256']
    cache = load_file(str(cache_path))
    assert set(cache) == {'input_ids', 'prompt_embeds'}
    assert cache['prompt_embeds'].shape == (1, 512, 4096) and cache['prompt_embeds'].dtype == torch.bfloat16
    assert torch.isfinite(cache['prompt_embeds']).all()
    author = ROOT / '.tools/ICTone-audit/inference.py'
    tree = ast.parse(author.read_bytes())
    prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(x, ast.Name) and x.id == 'DEFAULT_INSTANCE_PROMPT' for x in n.targets))
    assert prompt == report['prompt']
    tokenizer, identity = load_t5_tokenizer_from_ascii_cache(ASSETS / 'models/components/tokenizer_2')
    ids = tokenizer(prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt').input_ids
    torch.testing.assert_close(ids, cache['input_ids'], rtol=0, atol=0)
    assert identity == report['tokenizer_content_id'] and int((ids != tokenizer.pad_token_id).sum()) == 65
    result = dict(finished=True, result_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  tensor_payloads_verified=219, cache_verified=True, source_and_tokenizer_verified=True,
                  scope='file/record/cache audit only; repeated GPU equality is producer-observed, not independently rerun',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
