"""Fetch a frozen, byte-bounded subset of the public S24 derivative release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import ProxyHandler, Request, build_opener


REPO = 'https://huggingface.co/datasets/Lukhausen/S24-Raw-sRGB-2k'
API = 'https://huggingface.co/api/datasets/Lukhausen/S24-Raw-sRGB-2k'
REVISION = 'c233b3ab441d24b8fd57c33b0ab99a205fc8b84b'
MIB = 1024 ** 2
ROLES = {'data': '.json', 'srgb_images_style_1': '.jpg', 'srgb_images_style_2': '.jpg',
         'srgb_images_style_3': '.jpg', 'masks': '.png'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def request(opener, url, maximum):
    with opener.open(Request(url, headers={'Accept-Encoding': 'identity'}), timeout=22) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('S24 response exceeds its frozen byte budget')
    return raw


def fetch(out, proxy):
    opener = build_opener(ProxyHandler({'https': proxy})) if proxy else build_opener()
    out.mkdir(exist_ok=False)

    def listing(role):
        url = f'{API}/tree/{REVISION}/train/{role}?limit=30'
        raw = request(opener, url, 32768)
        rows = json.loads(raw)
        expected = {f'train/{role}/img_{i:04d}{ROLES[role]}' for i in range(30)}
        if len(rows) != 30 or set(row['path'] for row in rows) != expected:
            raise ValueError('Fixed list of 30 scenes changed')
        return role, (sha(raw), {PurePosixPath(r['path']).stem: r for r in rows})

    with ThreadPoolExecutor(max_workers=5) as pool:
        index = dict(pool.map(listing, ROLES))
    ids = sorted(index['data'][1], key=lambda identifier: sha(identifier.encode('ascii')))[:6]
    roles = [(identifier, role, index[role][1][identifier]) for identifier in ids for role in ROLES]
    total = sum(row['size'] for _, _, row in roles)
    if total > 32 * MIB or any(not 0 < row['size'] <= 3 * MIB for _, _, row in roles):
        raise ValueError('Fixed S24 media exceeds its byte budget')
    plan = {'revision': REVISION, 'ids': ids, 'role_tree_hashes': {k: v[0] for k, v in index.items()},
            'members': [r for _, _, r in roles], 'bytes': total,
            'protocol_sha256': sha(Path(__file__).with_name('s24-calibrated-look-pilot-protocol.md').read_bytes())}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    for identifier in ids:
        (out / identifier).mkdir()

    def get(job):
        identifier, role, row = job
        raw = request(opener, f'{REPO}/resolve/{REVISION}/{row["path"]}', row['size'])
        if len(raw) != row['size']:
            raise ValueError('Unexpected S24 file length')
        if 'lfs' in row:
            if sha(raw) != row['lfs']['oid']:
                raise ValueError('S24 LFS object changed')
        elif hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != row['oid']:
            raise ValueError('S24 Git blob changed')
        path = out / identifier / (role + ROLES[role])
        path.write_bytes(raw)
        return identifier, role, dict(sha256=sha(raw), bytes=len(raw))

    files = {identifier: {} for identifier in ids}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for identifier, role, value in pool.map(get, roles):
            files[identifier][role] = value
    (out / 'assets.json').write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(total_bytes=total, verified_files=len(roles), scenes=len(ids))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy')
    args = parser.parse_args()
    fetch(args.out, args.proxy)
