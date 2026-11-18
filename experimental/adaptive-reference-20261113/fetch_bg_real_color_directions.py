"""Fetch a fixed BG-REAL fitting split, separate from the four development pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

from fetch_bg_real_pixel_pilot import REVISION, fetch
from independent_coco import ROOT, digest

DOCS = ROOT / 'docs/research'
INTAKE_SHA = '79580687384ed0791faa8e1ef9b4926e29c3e8cf8a1b470f11fcd84ee3663a8c'
DEV_SHA = '83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9'


def read_pinned(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Unexpected source identity: {path.name}')
    return json.loads(path.read_bytes())


def select():
    intake = read_pinned(DOCS / '2026-11-18-bg-real-control-intake-evidence.json', INTAKE_SHA)
    development = read_pinned(DOCS / '2026-11-18-bg-real-pixel-pilot-evidence.json', DEV_SHA)
    pairs = intake['pairs']
    assert intake['revision'] == REVISION and len(pairs) == 87
    used = set(development['cases'])
    used |= {pairs[key]['donor'] for key in used}
    selection = []
    for key, row in sorted(pairs.items()):
        if row['subject'] in ('Man', 'Woman') and not {key, row['donor']} & used:
            selection.append((key, row))
            used.update((key, row['donor']))
        if len(selection) == 12:
            break
    assert len(selection) == 12
    return intake, development, selection


def run(attribution, out, proxy):
    intake, development, selection = select()
    if digest(attribution) != intake['files']['attribution.csv']['sha256']:
        raise ValueError('Attribution does not match the release')
    with attribution.open(encoding='utf-8-sig', newline='') as stream:
        credits = {row['bgreal_id']: row for row in csv.DictReader(stream)}
    out.mkdir(exist_ok=True)
    tasks = [(key, role, arm['path'], arm['sha256'])
             for key, row in selection for role, arm in row['arms'].items()]
    tasks += [(key, 'mask', row['subject_mask'], None) for key, row in selection]

    def download(task):
        key, role, source, expected = task
        data, meta = fetch(source, expected, proxy)
        path = out / f'{key}-{role}{Path(source).suffix}'
        if path.exists():
            assert path.read_bytes() == data
        else:
            with path.open('xb') as handle:
                handle.write(data)
        return key, role, dict(meta, name=path.name)

    members = {key: {} for key, _ in selection}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, role, meta in pool.map(download, tasks):
            members[key][role] = meta
    hashes = [item['sha256'] for row in members.values() for item in row.values()]
    held = {item['sha256'] for row in development['cases'].values() for item in row['input'].values()}
    assert len(hashes) == len(set(hashes)) == 36 and not set(hashes) & held
    attribution_rows = {key: [credits[key], credits[row['donor']]] for key, row in selection}
    for name, value in [('intake.json', members), ('attribution.json', attribution_rows)]:
        path = out / name
        data = (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
        if path.exists():
            assert path.read_bytes() == data
        else:
            with path.open('xb') as handle:
                handle.write(data)
    print(sum(item['size'] for row in members.values() for item in row.values()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attribution', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    run(args.attribution, args.out, args.proxy)
