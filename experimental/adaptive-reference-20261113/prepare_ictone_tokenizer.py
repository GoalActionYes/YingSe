"""Two pinned tokenizer dependencies, isolated and without pip dependency resolution."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / '.tools/ICTone-quantized-audit'


def main():
    manifest_path = TARGET / 'tokenizer-dependencies.json'
    assert not manifest_path.exists(), 'already prepared; verify existing manifest instead'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7897'}))
    specs = [('sentencepiece', '0.2.1', 'sentencepiece-0.2.1-cp312-cp312-win_amd64.whl',
              '4cdc7c36234fda305e85c32949c5211faaf8dd886096c7cea289ddc12a2d02de'),
             ('protobuf', '5.29.5', 'protobuf-5.29.5-cp310-abi3-win_amd64.whl',
              '3f76e3a3675b4a4d867b52e4a5f5b78a2ef9565549d4037e06cf7b0942b1d3fc')]
    records = []
    total = 0
    for name, version, filename, sha in specs:
        with opener.open(f'https://pypi.org/pypi/{name}/{version}/json', timeout=30) as response:
            raw = response.read(2_000_001)
        assert len(raw) <= 2_000_000
        metadata = json.loads(raw)
        wheel = next(x for x in metadata['urls'] if x['filename'] == filename)
        total += wheel['size']
        assert total <= 2_000_000 and wheel['digests']['sha256'] == sha
        with opener.open(wheel['url'], timeout=30) as response:
            data = response.read(wheel['size'] + 1)
        assert len(data) == wheel['size'] and hashlib.sha256(data).hexdigest() == sha
        (TARGET / filename).write_bytes(data)
        files = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                dest = (TARGET / 'vendor' / item.filename).resolve()
                assert dest.is_relative_to((TARGET / 'vendor').resolve()) and not dest.exists()
                body = archive.read(item)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
                files[item.filename] = hashlib.sha256(body).hexdigest()
        records.append(dict(package=name, version=version, filename=filename, bytes=len(data), sha256=sha, files=files))
    manifest_path.write_text(json.dumps(dict(finished=True, total_bytes=total, records=records), indent=2), encoding='utf-8')
    print(json.dumps(dict(finished=True, bytes=total)))


if __name__ == '__main__':
    main()
