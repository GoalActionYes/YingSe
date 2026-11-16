"""Bounded public metadata/config probes; never fetch weights or send images."""
import argparse
import ctypes
import datetime
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import get_token
from independent_coco import ROOT, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy')
    parser.add_argument('--run', default='joint-model-resource-audit-v1')
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': args.proxy})) if args.proxy else urllib.request.build_opener()

    def read(url):
        request = urllib.request.Request(url, headers={'User-Agent': 'YingSe-readonly-resource-audit'})
        try:
            with opener.open(request, timeout=30) as response:
                status, data = response.status, response.read(2_000_001)
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read(4096)
        assert len(data) <= 2_000_000
        return status, data

    report = dict(scope='Resource feasibility only; no model inference, images, training or weight downloads',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  local_hf_credential_available=bool(get_token()), sources=[], models={})
    for label, model in [('flux', 'black-forest-labs/FLUX.1-Fill-dev'), ('lora', 'ToneStyle/ICTone-Fill-LoRA'),
                         ('sage', 'chenxib/sage-color'), ('sd35', 'stabilityai/stable-diffusion-3.5-medium')]:
        url = f'https://huggingface.co/api/models/{model}?blobs=true'
        status, data = read(url)
        assert status == 200, (label, status)
        (out / (label + '-metadata.json')).write_bytes(data)
        metadata = json.loads(data)
        files = []
        for item in metadata['siblings']:
            name = item['rfilename']
            selected = (label == 'flux' and name.startswith(('transformer/', 'vae/', 'text_encoder/', 'text_encoder_2/')) and name.endswith('.safetensors')
                        or label == 'lora' and name.endswith('.safetensors')
                        or label == 'sage' and name == 'checkpoints/sage-color-final.pt')
            if selected:
                assert item['size'] == item['lfs']['size']
                files.append(dict(path=name, bytes=item['size'], sha256=item['lfs']['sha256']))
        entry = dict(repository=model, revision=metadata['sha'], gated=metadata.get('gated'),
                     selected_weights=files, selected_weight_bytes=sum(x['bytes'] for x in files),
                     metadata_sha256=hashlib.sha256(data).hexdigest())
        if label in ['flux', 'sd35']:
            config_url = f"https://huggingface.co/{model}/resolve/{metadata['sha']}/transformer/config.json"
            config_status, config_data = read(config_url)
            entry['unauthenticated_config'] = dict(status=config_status, response_bytes=len(config_data), url=config_url)
        report['models'][label] = entry
    commit_url = 'https://api.github.com/repos/dengyuhai/ICTone/commits/main'
    status, raw = read(commit_url)
    assert status == 200
    revision = json.loads(raw)['sha']
    url = f'https://raw.githubusercontent.com/dengyuhai/ICTone/{revision}/inference.py'
    status, code = read(url)
    assert status == 200
    (out / 'author-inference.py').write_bytes(code)
    report['author_inference'] = dict(revision=revision, url=url, sha256=hashlib.sha256(code).hexdigest(),
                                    model_cpu_offload_call=b'pipe.enable_model_cpu_offload()' in code,
                                    sequential_cpu_offload_call=b'enable_sequential_cpu_offload(' in code,
                                    quantization_config=b'quantization_config' in code)
    assert report['models']['flux']['selected_weights'] and report['models']['lora']['selected_weights']
    report['ictone_selected_weight_bytes'] = sum(report['models'][k]['selected_weight_bytes'] for k in ['flux', 'lora'])
    report['flux_transformer_weight_bytes'] = sum(x['bytes'] for x in report['models']['flux']['selected_weights'] if x['path'].startswith('transformer/'))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    report['gpu_csv_mib'] = gpu.stdout.strip()

    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ['total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended']]

    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    report['ram_bytes'] = dict(total=memory.total_phys, available=memory.avail_phys)
    report['source_sha256'] = digest(Path(__file__))
    report['finished'] = True
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    public = ROOT / 'docs/research/2026-11-17-joint-model-resource-evidence.json'
    public.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['models', 'sources']}, indent=2))


if __name__ == '__main__':
    main()
