"""Check public PST50 input identity without generating previews or grading outputs."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

from coverage_duplicates import perceptual_hash, pixel_digest
from independent_coco import ROOT, digest
from reference_color_app.media import decode_image


BASE = ROOT / 'outputs/adaptive-reference-20261113'
REV = '6ed86e34c37c558a0b2401d52f06603d619472e1'
ROLES = ('content_709', 'paired_style', 'paired_gt')
PINNED = {
    'v1': 'd723d66c96c69aced1c71c3bd626f03e30ae069e76fc3470cd439d60f4e30b88',
    'v2': 'da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, sha):
    if not path.is_file() or digest(path) != sha:
        raise ValueError(f'Unexpected file identity: {path.name}')


def load_entries():
    report = read_json(ROOT / 'docs/research/2026-11-18-pst50-reserve-audit-evidence.json')
    if report['revision'] != REV:
        raise ValueError('Unexpected PST50 source revision')
    roles = {}
    for role in ROLES:
        path = BASE / 'pst50-reserve-metadata-v1' / f'{role}.json'
        verify(path, report['roles'][role]['metadata_sha256'])
        rows = read_json(path)
        entries = {}
        for row in rows:
            match = re.fullmatch(fr'{role}/[a-z]+([0-9]+)\.png', row['path'])
            if not match or int(match[1]) in entries or row['size'] != row['lfs']['size']:
                raise ValueError('Unexpected PST50 path or size')
            entries[int(match[1])] = row
        if sorted(entries) != list(range(1, 51)):
            raise ValueError('Unexpected PST50 numbering')
        roles[role] = entries
    return roles


def fetch(out, row, proxy):
    path = out / 'inputs' / row['path']
    expected = row['lfs']['oid']
    if path.exists():
        verify(path, expected)
        return path
    tmp = path.with_suffix('.download')
    url = f'https://huggingface.co/datasets/zrgong/PST50/resolve/{REV}/{quote(row["path"], safe="/")}'
    command = ['curl.exe', '--fail', '--silent', '--show-error', '--location', '--max-time', '65',
               '--max-filesize', str(row['size']), '--output', str(tmp)]
    if proxy:
        command.extend(('--proxy', proxy))
    subprocess.run([*command, url], check=True)
    verify(tmp, expected)
    if tmp.stat().st_size != row['size']:
        raise ValueError('Unexpected downloaded byte count')
    tmp.replace(path)
    return path


def fingerprint(path):
    rgb = decode_image(path)
    h, w = rgb.shape[:2]
    y, x = h // 20, w // 20
    center = rgb[y:h-y, x:w-x]
    return dict(bytes_sha256=digest(path), pixels_sha256=pixel_digest(rgb), shape=[h, w],
                fingerprints=[perceptual_hash(value) for value in (rgb, rgb[:, ::-1], center, center[:, ::-1])])


def local_consumed(entries):
    files = {}
    for role, rows in entries.items():
        for index in range(1, 16):
            row = rows[index]
            folder = ('pst50' if index <= 5 else 'pst50/development-dense-round2-v1/inputs')
            path = BASE / folder / row['path']
            verify(path, row['lfs']['oid'])
            files[f'PST50 consumed/{index}/{role}'] = path
    natural = BASE / 'tst100k/reserved-multiref-v1'
    rows = read_json(natural / 'results.json')['cases']
    if len(rows) != 32:
        raise ValueError('Unexpected natural development cohort')
    for row in rows:
        files[f'Natural/{row["index"]}/input'] = natural / row['outputs']['input']['path']
        files[f'Natural/{row["index"]}/reference'] = natural / 'srgb' / (row['triplet']['reference'] + '.png')
    for path in sorted((BASE / 'independent-coco').glob('*-input.png')):
        files[f'COCO/{path.name}'] = path
    for path in sorted((BASE / 'facd-range-media-v1').glob('*/*.jpg')):
        files[f'FACD/{path.parent.name}/{path.name}'] = path
    for name, folder in [(102, 'psbench-shared-local-v1'), (105, 'psbench-shared-local-v1'),
                         (230, 'psbench-local-source-v1')]:
        for suffix in ('start', 'end'):
            files[f'PSBench/{name}_{suffix}'] = BASE / folder / f'{name}_{suffix}.jpg'
    return {name: fingerprint(path) for name, path in files.items()}


def cached_consumed():
    index = {}
    for name, sha in PINNED.items():
        path = BASE / 'tst100k' / f'cohort-training-coverage-{name}' / 'srgb-v1/manifest.json'
        verify(path, sha)
        data = read_json(path)
        for role, key, hash_key in [('files', 'original_path', 'perceptual_hash'), ('heldout', 'path', 'hash')]:
            for row in data[role]:
                index[row['original_sha256']] = (f'TST {name}/{row[key]}', row[hash_key], row['pixel_sha256'])
    return index


def distance(a, b):
    return min((x ^ y).bit_count() for x in a for y in b)


def run(out, proxy):
    entries = load_entries()
    jobs = [(index, role, entries[role][index]) for index in range(16, 51) for role in ROLES[:2]]
    if any(row['size'] > 7 * 1024 ** 2 for _, _, row in jobs) or sum(j[2]['size'] for j in jobs) != 222_508_285:
        raise ValueError('Unexpected byte budget')
    out.mkdir(exist_ok=True)
    if (out / 'results.json').exists():
        raise FileExistsError('Existing results are immutable')
    for role in ROLES[:2]:
        (out / 'inputs' / role).mkdir(parents=True, exist_ok=True)
    local = local_consumed(entries)
    cached = cached_consumed()
    reserve = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        downloads = pool.map(lambda job: fetch(out, job[2], proxy), jobs)
        for (index, role, _), path in zip(jobs, downloads, strict=True):
            reserve.setdefault(index, {})[role] = fingerprint(path)
            if role == ROLES[1]:
                print(f'Checked input bytes for group {index}', flush=True)
    within = []
    cross = []
    links = {index: {index} for index in reserve}
    for (i, a), (j, b) in combinations(reserve.items(), 2):
        for role_a, first in a.items():
            for role_b, second in b.items():
                gap = distance(first['fingerprints'], second['fingerprints'])
                if gap <= 6 or first['pixels_sha256'] == second['pixels_sha256']:
                    within.append([i, role_a, j, role_b, gap])
                    links[i].add(j)
                    links[j].add(i)
    for i, images in reserve.items():
        for role, f in images.items():
            for name, other in local.items():
                gap = distance(f['fingerprints'], other['fingerprints'])
                if gap <= 6 or f['pixels_sha256'] == other['pixels_sha256']:
                    cross.append([i, role, name, gap])
            for sha, (name, value, pixels) in cached.items():
                gap = distance(f['fingerprints'], [value])
                if gap <= 6 or f['bytes_sha256'] == sha or f['pixels_sha256'] == pixels:
                    cross.append([i, role, name, gap])
    seen = set()
    components = []
    for seed in reserve:
        if seed in seen:
            continue
        pending = [seed]
        component = {seed}
        while pending:
            unseen = links[pending.pop()] - component
            component.update(unseen)
            pending.extend(sorted(unseen))
        seen.update(component)
        components.append(sorted(component))
    direct = {record[0] for record in cross}
    isolated = sorted(i for group in components if direct.intersection(group) for i in group)
    result = dict(revision=REV, verified_reserved_files=len(jobs), reserved_bytes=sum(j[2]['size'] for j in jobs),
                  consumed_local_count=len(local), consumed_cached_count=len(cached),
                  within=within, cross=cross, components=components, isolated=isolated,
                  unresolved=[i for i in reserve if i not in isolated], fingerprints=reserve,
                  scope_counts={key: sum(name.startswith(key + '/') for name in local)
                                for key in ('PST50 consumed', 'Natural', 'COCO', 'FACD', 'PSBench')},
                  consumed_manifests=PINNED, script_sha256=digest(__file__),
                  protocol_sha256=digest(Path(__file__).with_name('pst50-reserve-fingerprint-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('verified_reserved_files', 'scope_counts', 'consumed_cached_count',
                                                   'isolated', 'unresolved', 'components')}))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    run(**vars(parser.parse_args()))
