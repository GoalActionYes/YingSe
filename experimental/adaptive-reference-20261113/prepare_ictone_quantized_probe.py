"""Install two hash-verified pure Python wheels into an isolated probe vendor."""
import hashlib
import importlib.metadata
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    target = ROOT / '.tools/ICTone-quantized-audit'
    vendor = target / 'vendor'
    vendor.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'}))
    manifest = dict(packages=[], unchanged_environment={})
    total = 0
    for name, version, expected in [
        ('gguf', '0.17.1', '7bc5aa7eeb1931f7d39b48fdc5b38fda6b294b9dca75cf607ac69557840a3943'),
        ('peft', '0.18.0', '624f69ca6393b765ccc6734adda7ca57d80b238f0900a42c357d8b67a03d62ff'),
    ]:
        with opener.open(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as response:
            raw = response.read(2_000_001)
        assert len(raw) <= 2_000_000
        metadata = json.loads(raw)
        wheel = next(x for x in metadata['urls'] if x['filename'].endswith('none-any.whl'))
        assert wheel['digests']['sha256'] == expected
        total += wheel['size']
        assert total <= 5_000_000
        with opener.open(wheel['url'], timeout=30) as response:
            data = response.read(wheel['size'] + 1)
        assert len(data) == wheel['size'] and hashlib.sha256(data).hexdigest() == expected
        (target / wheel['filename']).write_bytes(data)
        files = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for item in archive.infolist():
                destination = (vendor / item.filename).resolve()
                assert destination.is_relative_to(vendor.resolve())
                if not item.is_dir():
                    body = archive.read(item)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(body)
                    files[item.filename] = hashlib.sha256(body).hexdigest()
        manifest['packages'].append(dict(name=name, version=version, filename=wheel['filename'],
                                         bytes=len(data), sha256=expected, files=files,
                                         requirements=metadata['info']['requires_dist']))
    for name in ['torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'huggingface-hub']:
        manifest['unchanged_environment'][name] = importlib.metadata.version(name)
    manifest['total_download_bytes'] = total
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(bytes=total, wheels=[x['filename'] for x in manifest['packages']])))


if __name__ == '__main__':
    main()
