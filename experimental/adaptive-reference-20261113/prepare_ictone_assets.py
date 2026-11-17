"""Pinned bounded asset plan and resumable downloads for offline research only."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / '.tools/ICTone-quantized-audit'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1_048_576), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(path, item):
    assert path.stat().st_size == item['bytes']
    sha = digest(path)
    if item['sha256'] is not None:
        assert sha == item['sha256']
    else:
        assert item['bytes'] <= 5_000_000 and item.get('git_blob_sha1')
        blob = hashlib.sha1(b'blob ' + str(item['bytes']).encode() + b'\0' + path.read_bytes()).hexdigest()
        assert blob == item['git_blob_sha1']
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--only', choices=['t5', 'fill', 'lora', 'components', 'all'], default='all')
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'}))
    previous = json.loads((ROOT / 'outputs/adaptive-reference-20261113/ictone-quantized-metadata-v1/results.json').read_text())
    original = json.loads((ROOT / 'docs/research/2026-11-17-joint-model-resource-evidence.json').read_text())
    metadata_path = TARGET / 'component-metadata/weights--flux-fill-dev.json'
    revision = 'e27af7813f95c5f0e3e7ebc623f10681f9f09a34'
    if not metadata_path.exists():
        with opener.open(f'https://huggingface.co/api/models/weights/flux-fill-dev/revision/{revision}?blobs=true', timeout=30) as response:
            raw = response.read(2_000_001)
        assert len(raw) <= 2_000_000 and json.loads(raw)['sha'] == revision
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(raw)
    metadata = json.loads(metadata_path.read_text())
    assert metadata['sha'] == revision
    assets = []
    for label, suffix in [('t5', 'Q3_K_S.gguf'), ('fill', 'Q3_K_S.gguf'), ('lora', '.safetensors')]:
        model = previous['models'][label]
        entry = next(x for x in model['files'] if x['path'].endswith(suffix))
        assets.append(dict(label=label, repository=model['repository'], revision=model['revision'],
                           path=entry['path'], bytes=entry['bytes'], sha256=entry['lfs']['sha256'],
                           target='models/' + label + '/' + entry['path']))
    expected = {x['path']: x for x in original['models']['flux']['selected_weights']}
    for item in metadata['siblings']:
        path = item['rfilename']
        selected = (path.startswith(('tokenizer/', 'tokenizer_2/')) or path in [
            'model_index.json', 'scheduler/scheduler_config.json', 'transformer/config.json',
            'text_encoder/config.json', 'text_encoder/model.safetensors',
            'vae/config.json', 'vae/diffusion_pytorch_model.safetensors'])
        if not selected:
            continue
        if path in expected:
            assert item['lfs']['sha256'] == expected[path]['sha256'] and item['size'] == expected[path]['bytes']
        assets.append(dict(label='components', repository='weights/flux-fill-dev', revision=metadata['sha'],
                           path=path, bytes=item['size'], sha256=item.get('lfs', {}).get('sha256'),
                           git_blob_sha1=item.get('blobId'), target='models/components/' + path))
    total = sum(x['bytes'] for x in assets)
    assert total <= 10_000_000_000 and shutil.disk_usage(TARGET).free >= 30_000_000_000
    plan = dict(scope='offline noncommercial research only; no product integration permission inferred',
                assets=assets, total_bytes=total, budget_bytes=10_000_000_000,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plan_path = TARGET / 'assets-plan.json'
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text())
        assert {k:v for k,v in old_plan.items() if k != 'source_sha256'} == {k:v for k,v in plan.items() if k != 'source_sha256'}, 'asset plan differs; preserve old plan and review explicitly'
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(dict(plan_bytes=total, files=len(assets), download=args.download, only=args.only)), flush=True)
    if not args.download:
        return
    chosen = [x for x in assets if args.only == 'all' or x['label'] == args.only]
    deadline = time.monotonic() + 1800
    records = []
    for item in chosen:
        dest = TARGET / item['target']
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.partial')
        if dest.exists():
            sha = verify_asset(dest, item)
            records.append(dict(target=item['target'], bytes=item['bytes'], sha256=sha, reused=True))
            continue
        start = part.stat().st_size if part.exists() else 0
        assert start <= item['bytes']
        url = f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{item['path']}?download=true"
        while start < item['bytes']:
            assert time.monotonic() < deadline, '1800-second download budget exhausted; partial files preserved'
            # A new bounded range per request avoids retaining an expired signed
            # URL. Never log redirected URLs or credentials.
            end = min(start + 32 * 1_048_576, item['bytes']) - 1
            chunk_path = part.with_name(part.name + '.range')
            for attempt in range(3):
                try:
                    subprocess.run([sys._base_executable, str(Path(__file__).resolve()), '--range-worker',
                                    url + '&range_start=' + str(start), str(start), str(end), str(item['bytes']),
                                    str(chunk_path)], check=True, timeout=min(90, max(1, deadline-time.monotonic())),
                                   capture_output=True)
                    break
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    print(json.dumps(dict(file=item['target'], start=start, attempt=attempt+1,
                                          status='range_timeout_or_transport_error')), flush=True)
                    if attempt == 2:
                        raise RuntimeError('range failed three times; verified prefix preserved') from None
            assert chunk_path.stat().st_size == end - start + 1
            with chunk_path.open('rb') as source, part.open('ab') as handle:
                shutil.copyfileobj(source, handle, length=1_048_576)
            start = part.stat().st_size
            print(json.dumps(dict(file=item['target'], bytes=start, total=item['bytes'])), flush=True)
        sha = verify_asset(part, item)
        part.rename(dest)
        records.append(dict(target=item['target'], bytes=item['bytes'], sha256=sha, reused=False))
    (TARGET / ('assets-' + args.only + '-verified.json')).write_text(json.dumps(dict(finished=True, records=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=True, files=len(records), only=args.only)), flush=True)


def range_worker(url, start, end, total, destination):
    start, end, total = int(start), int(end), int(total)
    assert 0 <= start <= end < total and end-start < 32*1_048_576
    destination = Path(destination).resolve()
    assert destination.is_relative_to((TARGET / 'models').resolve()) and destination.name.endswith('.range')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    request = urllib.request.Request(url, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
    with opener.open(request, timeout=30) as response:
        if response.status == 206:
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
        else:
            assert response.status == 200 and start == 0 and end+1 == total
        remaining = end-start+1
        with destination.open('wb') as handle:
            while remaining:
                data = response.read(min(1_048_576, remaining))
                assert data
                handle.write(data)
                remaining -= len(data)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--range-worker':
        range_worker(*sys.argv[2:])
    else:
        main()
