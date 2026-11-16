"""Fetch only hash-pinned public source files; no weights, dependencies or media."""
import hashlib
import json
import urllib.request

from independent_coco import ROOT


def main():
    evidence = json.loads((ROOT / 'docs/research/2026-11-17-gradestudio-evidence.json').read_text())
    revision = evidence['revision']
    root = ROOT / '.tools/GradeStudio-source'
    root.mkdir(exist_ok=True)
    for item in evidence['author_file_hashes']:
        relative = item['path']
        assert not relative.startswith('/') and '..' not in relative.split('/')
        path = root / relative
        if path.exists():
            data = path.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/LivDevGit/GradeStudio/{revision}/{relative}'
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(1_000_001)
            assert len(data) <= 1_000_000
        assert hashlib.sha256(data).hexdigest() == item['sha256'], relative
        assert len(data) == item['bytes']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    manifest = root / 'source-audit.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        assert existing['revision'] == revision and existing['files'] == evidence['author_file_hashes']
    else:
        manifest.write_text(json.dumps(dict(revision=revision, files=evidence['author_file_hashes']), indent=2))
    print(f"Verified {len(evidence['author_file_hashes'])} source files at {revision}; no model downloads")


if __name__ == '__main__':
    main()
