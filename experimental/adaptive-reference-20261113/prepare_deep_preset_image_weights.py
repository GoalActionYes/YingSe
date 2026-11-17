"""Read the public legacy Deep Preset archive without its unused preset head."""
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path
import pickle
import re
import struct
import subprocess
import tempfile

import numpy as np
import torch


REVISION = '04eb47352e2f4cd643d466c7c6835f17d9513c92'
BLOBS = {
    '__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'blurpool.py': 'cbce4101eee9aa4aa38ce57bf712d3201d330e16',
    'components.py': '15a8eda83f2277ceb24adfea68f743a044ce8aac',
    'network.py': '5c031683af2224fb1241b07291b5b0fa795c1626',
    'norm.py': 'b0db30264ee3ae89883c8148a778b7a6dfb96e3e',
}
URL = 'https://drive.usercontent.google.com/download?id=1GegyHf3OD17k_WID3-vA7S8nRQwPfpTC&export=download&confirm=t'
SIZE = 1_071_225_835
HEAD_SHA = 'b36197cc2356eb60fac7be7c569ad042e4306019df3e480b1f87d3b25b954900'
OPTIONS = dict(g_depth=5, g_norm='evo', g_net='net', g_out_channels=[69, 3], g_in_channels=[6, 3],
               g_upsampler='up_blurbilinear', g_downsampler='down_blurmax')


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('argparse', 'Namespace'): Namespace, ('collections', 'OrderedDict'): OrderedDict,
                   ('torch', 'FloatStorage'): 'float32',
                   ('torch._utils', '_rebuild_tensor_v2'): lambda s, o, n, d, _g, _h: (s, o, n, d)}
        if (module, name) not in allowed:
            raise ValueError(f'Unexpected checkpoint global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, record):
        kind, dtype, ident, _device, count, view = record
        if kind != 'storage' or dtype != 'float32' or view is not None or type(count) is not int or count < 1:
            raise ValueError('Unexpected checkpoint storage metadata')
        return ident, count


def plan(metadata):
    if hashlib.sha256(metadata).hexdigest() != HEAD_SHA:
        raise ValueError('Frozen checkpoint prefix changed')
    stream = BytesIO(metadata)
    magic, version, info, saved, order = (MetadataReader(stream).load() for _ in range(5))
    if magic != 119547037146038801333356 or version != 1001 or not info['little_endian']:
        raise ValueError('Unexpected checkpoint format')
    if set(saved) != {'G', 'opts'} or vars(saved['opts']) != OPTIONS:
        raise ValueError('Unexpected checkpoint architecture')
    by_id = {}
    for name, (storage, offset, shape, stride) in saved['G'].items():
        ident, count = storage
        expected_stride = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
        if ident in by_id or offset or tuple(stride) != expected_stride or np.prod(shape) != count:
            raise ValueError(f'Unexpected tensor view: {name}')
        by_id[ident] = (name, count, shape)
    if len(order) != len(by_id) or set(order) != set(by_id):
        raise ValueError('Unexpected storage ordering')
    runs, position = [], stream.tell()
    for ident in order:
        name, count, shape = by_id[ident]
        end = position + 8 + 4 * count
        if not name.startswith('llayer_'):
            if not runs or runs[-1][1] != position:
                runs.append([position, position, []])
            runs[-1][1] = end
            runs[-1][2].append((name, count, shape))
        position = end
    if position != SIZE or len(runs) != 6 or sum(b - a for a, b, _ in runs) != 47_482_012:
        raise ValueError('Unexpected total checkpoint or image weight size')
    return runs


def fetch(url, limit, proxy, range_=None):
    with tempfile.TemporaryDirectory() as temp:
        headers = str(Path(temp) / 'headers.txt')
        command = ['curl.exe', '--fail', '--silent', '--show-error', '--location', '--max-time', '90',
                   '--max-filesize', str(limit), '--dump-header', headers]
        if proxy:
            command.extend(('--proxy', proxy))
        if range_:
            command.extend(('--range', f'{range_[0]}-{range_[1]-1}'))
        result = subprocess.run([*command, url], check=True, capture_output=True).stdout
        if range_:
            info = Path(headers).read_text('iso-8859-1').strip().lower().split('\n\n')[-1]
            value = re.search(r'^content-range: (.*)$', info, re.M)
            if (info.splitlines()[0].split()[1] != '206' or value is None or
                    value[1] != f'bytes {range_[0]}-{range_[1]-1}/{SIZE}' or len(result) != limit):
                raise ValueError('Unexpected byte-range response')
        return result


def run(metadata, out, proxy):
    ranges = plan(metadata.read_bytes())
    out.mkdir(exist_ok=False)
    network = out / 'networks'
    network.mkdir()
    sources = {}
    for name, expected in BLOBS.items():
        content = fetch(f'https://raw.githubusercontent.com/minhmanho/deep_preset/{REVISION}/networks/{name}',
                        16000, proxy)
        if hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() != expected:
            raise ValueError(f'Pinned Python source changed: {name}')
        (network / name).write_bytes(content)
        sources[name] = hashlib.sha256(content).hexdigest()
    parameters, receipts = {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(fetch, URL, end - begin, proxy, (begin, end)) for begin, end, _ in ranges]
        for (begin, end, entries), job in zip(ranges, jobs, strict=True):
            data = job.result()
            offset = 0
            for name, count, shape in entries:
                if struct.unpack_from('<q', data, offset)[0] != count:
                    raise ValueError(f'Storage length mismatch: {name}')
                arr = np.frombuffer(data, dtype='<f4', count=count, offset=offset + 8).reshape(shape).copy()
                if not np.isfinite(arr).all():
                    raise ValueError(f'Nonfinite weights: {name}')
                parameters[name] = torch.from_numpy(arr)
                offset += 8 + 4 * count
            if offset != len(data):
                raise ValueError('Unexpected trailing tensor data')
            receipts.append(dict(start=begin, end_exclusive=end, sha256=hashlib.sha256(data).hexdigest()))
    weights = out / 'image-weights.pth'
    torch.save(parameters, weights)
    manifest = dict(revision=REVISION, sources=sources, ranges=receipts, prefix_sha256=HEAD_SHA, options=OPTIONS,
                    image_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), tensors=len(parameters))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(tensors=len(parameters), bytes=weights.stat().st_size)))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    run(**vars(parser.parse_args()))
