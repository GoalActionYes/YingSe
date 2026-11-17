"""Bounded metadata/header audit. Never downloads full model payloads."""
import argparse
import collections
import datetime
import hashlib
import io
import json
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 2_000_000


class HeaderReader:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def read(self, n):
        assert 0 <= n <= LIMIT
        value = self.stream.read(n)
        if len(value) != n:
            raise ValueError('bounded header is incomplete')
        return value

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self):
        return self.read(self.unpack('Q')).decode('utf-8')

    def value(self, kind):
        scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in scalar:
            return self.unpack(scalar[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            subkind, count = self.unpack('I'), self.unpack('Q')
            # Tokenizer precompiled_charsmap can validly exceed 100k bytes.
            # Bound by the entire prefix budget; read() still enforces EOF.
            assert count <= LIMIT
            return [self.value(subkind) for _ in range(count)]
        raise ValueError(f'unsupported GGUF metadata type {kind}')


def parse_gguf(data):
    r = HeaderReader(data)
    assert r.read(4) == b'GGUF'
    version, count, kv_count = r.unpack('I'), r.unpack('Q'), r.unpack('Q')
    assert version in (2, 3) and count <= 100_000 and kv_count <= 10_000
    metadata = {}
    for _ in range(kv_count):
        key = r.string()
        metadata[key] = r.value(r.unpack('I'))
    tensors = []
    for _ in range(count):
        name, ndim = r.string(), r.unpack('I')
        assert ndim <= 8
        tensors.append(dict(name=name, ggml_dims=[r.unpack('Q') for _ in range(ndim)],
                            ggml_type=r.unpack('I'), offset=r.unpack('Q')))
    return dict(version=version, metadata=metadata, tensors=tensors,
                parsed_header_bytes=r.stream.tell(), payload_verified=False)


def parse_safetensors(data):
    length = struct.unpack('<Q', data[:8])[0]
    assert length <= LIMIT - 8 and len(data) >= length + 8
    header = json.loads(data[8:8 + length])
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    dtypes = collections.Counter(v['dtype'] for v in tensors.values())
    elements = collections.Counter()
    for v in tensors.values():
        n = 1
        for dim in v['shape']:
            n *= dim
        elements[v['dtype']] += n
    return dict(header_bytes=length + 8, metadata=header.get('__metadata__'), tensors=tensors,
                dtype_tensor_counts=dict(dtypes), dtype_elements=dict(elements), payload_verified=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--proxy', default='http://127.0.0.1:7897')
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': args.proxy}))

    def request(url, filename, header=False):
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'YingSe-bounded-format-audit'}
        if header:
            headers['Range'] = f'bytes=0-{LIMIT - 1}'
        with opener.open(urllib.request.Request(url, headers=headers), timeout=30) as response:
            data = response.read(LIMIT if header else LIMIT + 1)
            status, content_range = response.status, response.headers.get('Content-Range')
        assert len(data) <= LIMIT
        (out / filename).write_bytes(data)
        sources.append(dict(url=url, status=status, bytes_read=len(data), content_range=content_range,
                            sha256=hashlib.sha256(data).hexdigest(), scope='prefix_only' if header else 'complete_metadata'))
        return data

    sources, models = [], {}
    for label, repo in [('fill', 'YarvixPA/FLUX.1-Fill-dev-GGUF'),
                        ('lora', 'ToneStyle/ICTone-Fill-LoRA'),
                        ('t5', 'city96/t5-v1_1-xxl-encoder-gguf')]:
        raw = request(f'https://huggingface.co/api/models/{repo}?blobs=true', label + '-metadata.json')
        m = json.loads(raw)
        models[label] = dict(repository=repo, revision=m['sha'], gated=m.get('gated'),
                             license=m.get('cardData', {}).get('license'),
                             files=[dict(path=x['rfilename'], bytes=x.get('size'), lfs=x.get('lfs')) for x in m['siblings']])
    fill = next(x for x in models['fill']['files'] if x['path'].lower().endswith('q3_k_s.gguf'))
    lora = next(x for x in models['lora']['files'] if x['path'].endswith('.safetensors'))
    parsed = {}
    for label, entry, parser in [('fill', fill, parse_gguf), ('lora', lora, parse_safetensors)]:
        model = models[label]
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{entry['path']}"
        parsed[label] = parser(request(url, label + '-header.bin', True))
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), scope='metadata and bounded headers only',
                  sources=sources, models=models, parsed=parsed,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), finished=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=True, output=str(out), fill_tensors=len(parsed['fill']['tensors']),
                         fill_inputs=[x for x in parsed['fill']['tensors'] if 'img_in' in x['name']],
                         lora_dtypes=parsed['lora']['dtype_elements'], lora_sample=list(parsed['lora']['tensors'].items())[:4]), indent=2))


if __name__ == '__main__':
    main()
