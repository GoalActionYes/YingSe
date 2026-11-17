"""Read a bounded pinned T5 header and match all tensor shapes on meta."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from audit_ictone_quantized import ROOT, LIMIT, parse_gguf

sys.path[:0] = [str(ROOT / '.tools/ICTone-quantized-audit/vendor'),
                str(ROOT / '.tools/StyleGallery-audit/vendor'), str(ROOT / '.tools/ModFlows-audit/vendor')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--prefix', help='Reuse an already saved bounded prefix instead of fetching it again')
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    previous = json.loads(Path(args.metadata).read_text(encoding='utf-8'))['models']['t5']
    entry = next(x for x in previous['files'] if x['path'].endswith('Q3_K_S.gguf'))
    url = f"https://huggingface.co/{previous['repository']}/resolve/{previous['revision']}/{entry['path']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'}))
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{LIMIT-1}', 'Accept-Encoding': 'identity'})
    if args.prefix:
        prefix = Path(args.prefix)
        assert prefix.stat().st_size <= LIMIT
        raw = prefix.read_bytes()
        status, content_range = None, None
    else:
        with opener.open(request, timeout=30) as response:
            raw = response.read(LIMIT)
            status, content_range = response.status, response.headers.get('Content-Range')
    (out / 'header.bin').write_bytes(raw)
    header = parse_gguf(raw)
    (out / 'parsed-header.json').write_text(json.dumps(header, indent=2), encoding='utf-8')
    print(json.dumps(dict(metadata={k:v for k,v in header['metadata'].items() if not k.startswith('tokenizer.')}, tensors=len(header['tensors']),
                         sample=header['tensors'][:4]), indent=2), flush=True)
    import torch
    from accelerate import init_empty_weights
    from transformers import T5Config, T5EncoderModel
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map
    # This is the published T5-v1.1 XXL encoder architecture, verified below
    # against every header tensor rather than assumed compatible by filename.
    config = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240,
                      num_layers=24, num_heads=64, feed_forward_proj='gated-gelu',
                      is_encoder_decoder=False, use_cache=False)
    with init_empty_weights():
        model = T5EncoderModel(config)
    mapping = get_gguf_hf_weights_map(model, model_type='t5encoder', num_layers=24)
    state = model.state_dict()
    matched = []
    for item in header['tensors']:
        name = mapping[item['name']]
        shape = list(reversed(item['ggml_dims']))
        assert list(state[name].shape) == shape, (item['name'], name, shape, list(state[name].shape))
        matched.append(dict(gguf_name=item['name'], hf_name=name, shape=shape, ggml_type=item['ggml_type']))
    missing = set(state) - {x['hf_name'] for x in matched}
    # Tied embedding aliases share exactly the same tensor.
    assert missing <= {'shared.weight', 'encoder.embed_tokens.weight'}, missing
    assert any(x['hf_name'] in {'shared.weight', 'encoder.embed_tokens.weight'} for x in matched)
    result = dict(finished=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  repository=previous['repository'], revision=previous['revision'], file=entry,
                  url=url, status=status, content_range=content_range, bytes_read=len(raw),
                  prefix_sha256=hashlib.sha256(raw).hexdigest(), payload_verified=False,
                  reused_prefix=str(Path(args.prefix).resolve()) if args.prefix else None,
                  config=config.to_dict(), matched=matched, tied_aliases_not_separately_stored=sorted(missing))
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=True, matched=len(matched), tied_aliases=sorted(missing))))


if __name__ == '__main__':
    main()
