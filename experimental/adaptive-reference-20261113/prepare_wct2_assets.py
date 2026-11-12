"""Fetch pinned official WCT2 files; verify Git blob identity before any execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from independent_coco import ROOT, fetch, digest

COMMIT = '05d17c02c6eb78a3e0afdbc1cf1d7549d4ad1543'
FILES = ('model.py', 'transfer.py', 'utils/core.py', 'utils/io.py', 'README.md', 'requirements.txt',
         'model_checkpoints/wave_encoder_cat5_l4.pth', 'model_checkpoints/wave_decoder_cat5_l4.pth')


def main():
    from urllib.request import urlopen
    with urlopen(f'https://api.github.com/repos/clovaai/WCT2/git/trees/{COMMIT}?recursive=1', timeout=30) as response:
        tree = json.load(response)
    entries = {item['path']: item for item in tree['tree'] if item['path'] in FILES}
    if set(entries) != set(FILES):
        raise ValueError('Pinned source files missing')
    destination = ROOT / '.tools/WCT2-source'
    destination.mkdir(exist_ok=True)
    def prepare(relative):
        entry = entries[relative]
        path = destination / relative
        fetch(f'https://raw.githubusercontent.com/clovaai/WCT2/{COMMIT}/{relative}', path,
              size=entry['size'], maximum=40*1024*1024)
        raw = path.read_bytes()
        git_sha = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if git_sha != entry['sha']:
            raise ValueError('Source Git blob mismatch: '+relative)
        return {'path': relative, 'bytes': len(raw), 'git_blob_sha1': git_sha, 'sha256': digest(path)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(prepare, FILES))
    report = {'repository': 'https://github.com/clovaai/WCT2', 'commit': COMMIT,
              'license': 'Mixed: NAVER MIT-style notice; utils/core.py and utils/io.py carry NVIDIA CC-BY-NC-SA-4.0 notices. Research only; preserve all notices.',
              'files': files, 'scope': 'Assets only, not executed or integrated into product'}
    (destination / 'verified-assets.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
