"""Pinned public model assets only; resumable bounded HTTP, no remote Python."""
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from independent_coco import ROOT, digest

REPOSITORY = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
MODEL_SHA = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
FILES = ('config.json', 'generation_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json',
         'chat_template.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'README.md', 'model.safetensors')


def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError('Asset size mismatch: '+entry['path'])
    expected = entry.get('lfs', {}).get('oid')
    if expected:
        if digest(path) != expected:
            raise ValueError('Asset SHA256 mismatch: '+entry['path'])
    else:
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob != entry['oid']:
            raise ValueError('Asset Git blob mismatch: '+entry['path'])


def download(entry, destination):
    path = destination / entry['path']
    if path.exists():
        verify(path, entry)
        return
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry['size']:
            verify(partial, entry)
            partial.replace(path)
            return
        if offset > entry['size']:
            raise ValueError('Partial file exceeds declared size')
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{quote(entry["path"])}?download=true&offset={offset}'
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected resume range')
                    mode = 'ab'
                elif response.status == 200:
                    offset, mode = 0, 'wb'
                else:
                    raise ValueError('Unexpected download response')
                total, logged = offset, offset//(256*1024*1024)
                with partial.open(mode) as output:
                    while block := response.read(4*1024*1024):
                        total += len(block)
                        if total > entry['size']:
                            raise ValueError('Download exceeds declared size')
                        output.write(block)
                        if total//(256*1024*1024) > logged:
                            logged = total//(256*1024*1024)
                            print(json.dumps({'file': entry['path'], 'downloaded_MiB': total//(1024*1024)}), flush=True)
            if partial.stat().st_size != entry['size']:
                raise URLError('Incomplete response')
            verify(partial, entry)
            partial.replace(path)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            print(json.dumps({'file': entry['path'], 'attempt': attempt+1, 'retry_type': type(exc).__name__}), flush=True)
            if attempt == 4:
                raise RuntimeError('Download failed; partial retained for explicit resume') from None
            time.sleep(2)


def main():
    with urlopen(f'https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true', timeout=30) as response:
        tree = json.load(response)
    entries = {x['path']: x for x in tree if x['type'] == 'file' and x['path'] in FILES}
    if set(entries) != set(FILES) or entries['model.safetensors']['lfs']['oid'] != MODEL_SHA:
        raise ValueError('Pinned model asset set changed')
    destination = ROOT / '.tools/vision-advisor-model'
    destination.mkdir(exist_ok=True)
    records = []
    for name in FILES:
        entry = entries[name]
        download(entry, destination)
        records.append({'path': name, 'bytes': entry['size'], 'sha256': digest(destination / name)})
    license_url = 'https://www.apache.org/licenses/LICENSE-2.0.txt'
    with urlopen(license_url, timeout=30) as response:
        license_text = response.read(100000)
    if b'Apache License' not in license_text or b'Version 2.0' not in license_text:
        raise ValueError('Unexpected canonical license text')
    (destination / 'APACHE-2.0.txt').write_bytes(license_text)
    manifest = {'repository': REPOSITORY, 'revision': REVISION,
                'license': 'Apache-2.0 as upstream model-card metadata; card retained, canonical license text separately recorded',
                'license_url': license_url, 'license_text_sha256': hashlib.sha256(license_text).hexdigest(),
                'files': records, 'remote_python_downloaded': False, 'scope': 'Assets prepared only; no inference quality claim'}
    (destination / 'verified-assets.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'assets_verified': True, 'repository': REPOSITORY, 'revision': REVISION}), flush=True)


if __name__ == '__main__':
    main()
