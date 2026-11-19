"""Replay one recorded, drawn-mask color balance step on the author's raw."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from audit_playraw_sky_xmp import DT, RDF, payload


INPUT_HASHES = {
    'author.xmp': 'c4674014f48661e23f751842e604b33693b69d091a0ded746789aa006284496d',
    'P5020458.ORF': '7b4cc05a958acd8e10a161d1b2faf6cae9d25a37de28550b0680b34c14c38217',
    'darktable-cli.exe': '51865b25c91b2b936d249b8d8c71e6de58dcc7cdef7768ab9082781bc401d179',
}


def off_state(source):
    branches = list(ET.fromstring(source).iter(RDF + 'li'))
    active = [row for row in branches if row.get(DT + 'operation') == 'colorbalancergb']
    if [(r.get(DT + 'num'), r.get(DT + 'multi_priority')) for r in active] != [('27', '0'), ('28', '0')]:
        raise ValueError('Unexpected effective color balance instance')
    if int.from_bytes(payload(active[-1].get(DT + 'blendop_params'))[:4], 'little') != 3:
        raise ValueError('Expected a drawn mask')
    row, = [x for x in re.finditer(rb'<rdf:li\s+[^>]*>', source) if b'darktable:num="28"' in x[0]]
    before = b'darktable:enabled="1"'
    if row[0].count(before) != 1 or b'darktable:operation="colorbalancergb"' not in row[0]:
        raise ValueError('Unexpected step state')
    result = source[:row.start()] + row[0].replace(before, b'darktable:enabled="0"') + source[row.end():]
    if sum(a != b for a, b in zip(source, result, strict=True)) != 1:
        raise ValueError('More than one byte changed')
    return result


def export(cli, raw, xmp, out, label, mask):
    config, cache = out / f'config-{label}', out / f'cache-{label}'
    config.mkdir()
    cache.mkdir()
    destination = out / f'{label}.tif'
    args = [p.as_posix() for p in (cli, raw, xmp, destination)] + [
        '--width', '1024', '--height', '1024', '--hq', 'true', '--apply-custom-presets', 'false',
        '--icc-type', 'srgb', '--export_masks', str(mask).lower(), '--core', '--configdir', config.as_posix(),
        '--cachedir', cache.as_posix(), '--disable-opencl', '-t', '2', '--conf', 'write_sidecar_files=never',
        '--conf', 'plugins/imageio/format/tiff/bpp=8']
    with (out / f'{label}.log').open('wb') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True, cwd=cli.parent)
    return destination


def read(path, masked):
    with Image.open(path) as image:
        if image.mode != 'RGB' or max(image.size) != 1024 or not image.info.get('icc_profile'):
            raise ValueError('Unexpected export')
        rgb = np.array(image)
        mask = None
        if masked:
            for index in range(1, image.n_frames):
                image.seek(index)
                name = str(image.tag_v2.get(285, ''))
                try:
                    name = name.encode('latin1').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
                if name.casefold().strip() in {'color balance rgb', 'rgb 色彩平衡'}:
                    values = np.asarray(image)
                    if mask is not None or not np.array_equal(values, np.repeat(values[..., :1], 3, axis=-1)):
                        raise ValueError('Unexpected author mask page')
                    mask = values[..., 0].copy()
            if mask is None:
                raise ValueError('Missing author color balance mask')
    return rgb, mask


def run(source, out):
    cli = source / 'runtime/bin/darktable-cli.exe'
    raw, author = source / 'P5020458.ORF', source / 'author.xmp'
    if any(not p.is_absolute() or not p.as_posix().isascii() for p in (source, out)):
        raise ValueError('Use absolute ASCII paths')
    for path in (cli, raw, author):
        if sha256(path.read_bytes()).hexdigest() != INPUT_HASHES[path.name]:
            raise ValueError(f'Changed input: {path.name}')
    out.mkdir()
    off = out / 'off.xmp'
    off.write_bytes(off_state(author.read_bytes()))
    files = [export(cli, raw, xmp, out, label, mask)
             for xmp, label, mask in ((off, 'off', False), (author, 'on', True))]
    before, _ = read(files[0], False)
    after, mask = read(files[1], True)
    if before.shape != after.shape or before.shape[:2] != mask.shape:
        raise ValueError('Output geometry differs')
    delta = after.astype(np.float64) - before
    luma = delta @ np.array([.2126, .7152, .0722])
    chroma = np.abs(delta[..., [0, 2]] - luma[..., None]).mean(axis=-1)
    zones = {'inside': mask >= 204, 'outside': mask == 0}
    results = {}
    for label, region in zones.items():
        values = chroma[region]
        results[label] = {'pixels': len(values), 'median': float(np.median(values)) if len(values) else None,
                          'p95': float(np.quantile(values, .95)) if len(values) else None}
    passed = (min(v['pixels'] for v in results.values()) >= 128
              and results['inside']['median'] >= 2 and results['outside']['p95'] <= 1.5)
    record = {'shape': before.shape, 'regions': results, 'passed': passed,
              'mask_pixels_sha256': sha256(mask.tobytes()).hexdigest(),
              'rgb_sha256': [sha256(x.tobytes()).hexdigest() for x in (before, after)],
              'off_sha256': sha256(off.read_bytes()).hexdigest()}
    (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
