"""Download the same five development inputs in the author's native Log domain.

The paper/training dataset expects Log; upstream CLI README demonstrates Rec709.
Keep both tests separate rather than silently replacing an unfavorable result.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.request import urlopen
from urllib.parse import quote
from independent_coco import ROOT, fetch
from prepare_salut_assets import DATA


def main():
    with urlopen(f'https://huggingface.co/api/datasets/zrgong/PST50/tree/{DATA}/content_log', timeout=30) as response:
        listing = json.load(response)
    wanted = {f'content_log/in{i}.png' for i in range(1, 6)}
    entries = [entry for entry in listing if entry['path'] in wanted]
    if len(entries) != 5:
        raise ValueError('Expected exactly the same five development inputs')
    out = ROOT / 'outputs/adaptive-reference-20261113/pst50'
    def prepare(entry):
        sha, size = entry['lfs']['oid'], entry['size']
        fetch(f"https://huggingface.co/datasets/zrgong/PST50/resolve/{DATA}/{quote(entry['path'])}",
              out / entry['path'], sha, size, 30*1024*1024)
        return {'path': entry['path'], 'sha256': sha, 'bytes': size}
    with ThreadPoolExecutor(max_workers=3) as pool:
        assets = list(pool.map(prepare, entries))
    manifest = {'dataset_revision': DATA, 'development_pairs': list(range(1, 6)),
                'purpose': 'Native Log-domain control, not a solution for arbitrary RGB user input', 'assets': assets}
    (out / 'log-control-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
