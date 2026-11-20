"""Take fixed MIT room images by ZIP range and test a single-image spatial cue."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import re
import struct
import subprocess
from zipfile import ZipFile
import zlib

import numpy as np
from PIL import Image, ImageDraw

from probe_beyond_rgb_transfer import digest, save, write_json
from probe_multilum_gray_probe import apply, chroma, valid
from reference_color_app.media import decode_image


SCENES = ('elm_revis_kitchen14', 'everett_lobby1')
LIGHTS = (14, 24)
BASE_URL = 'https://data.csail.mit.edu/multilum'
LIMIT = 10 * 1024 ** 2
TAIL = 65536


def read_range(url, requested, limit, path, proxy):
    header = path.with_suffix('.headers')
    if not path.is_file() or not header.is_file():
        partial = path.with_suffix('.partial')
        subprocess.run(['curl.exe', '--proxy', proxy, '--silent', '--show-error', '--fail',
                        '--max-time', '35', '--range', requested, '--max-filesize', str(limit),
                        '--dump-header', str(header), '--output', str(partial), url], check=True)
        partial.replace(path)
    text = header.read_text()
    matches = re.findall(r'(?im)^content-range: bytes (\d+)-(\d+)/(\d+)\s*$', text)
    if not re.search(r'(?im)^HTTP/\S+\s+206\b', text) or len(matches) != 1:
        raise ValueError('Expected an exact 206 Content-Range')
    start, stop, total = map(int, matches[0])
    expected = (total - TAIL, total - 1) if requested == f'-{TAIL}' else tuple(map(int, requested.split('-')))
    data = path.read_bytes()
    if (start, stop) != expected or len(data) != stop - start + 1 or len(data) > limit:
        raise ValueError('Unexpected ZIP response length or range')
    return data, start, total


def intake(scene, folder, proxy):
    folder.mkdir(exist_ok=True)
    url = f'{BASE_URL}/{scene}/{scene}_mip2_jpg.zip'
    tail, origin, total = read_range(url, f'-{TAIL}', TAIL, folder / 'tail.range', proxy)
    results = {}
    remaining = LIMIT - len(tail)
    with ZipFile(BytesIO(tail)) as archive:
        for light in LIGHTS:
            member = f'{scene}/dir_{light}_mip2.jpg'
            item = archive.getinfo(member)
            start = item.header_offset + origin
            size = item.compress_size + 1024
            stop = start + size - 1
            if size > remaining or not (0 <= start < stop < total):
                raise ValueError('Fixed ZIP members exceed their request budget')
            raw, _, member_total = read_range(url, f'{start}-{stop}', size, folder / f'{light}.range', proxy)
            if member_total != total:
                raise ValueError('Archive changed during transfer')
            remaining -= len(raw)
            header = struct.unpack_from('<4s5H3I2H', raw)
            length, extra = header[-2:]
            begin = 30 + length + extra
            if (header[0] != b'PK\x03\x04' or header[2] & 1 or header[3] != item.compress_type
                    or raw[30:30 + length].decode() != member or begin > 1024):
                raise ValueError('Unexpected ZIP member header')
            data = raw[begin:begin + item.compress_size]
            image = zlib.decompress(data, -15) if item.compress_type == 8 else data
            if len(image) != item.file_size or zlib.crc32(image) != item.CRC:
                raise ValueError('ZIP member failed integrity checks')
            path = folder / f'{light}.jpg'
            path.write_bytes(image)
            results[light] = dict(path=path, sha256=digest(path), range=[start, stop])
    return results, dict(archive_bytes=total, bytes_read=LIMIT-remaining, zip_tail_sha256=digest(folder / 'tail.range'))


def spatial_dispersion(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((512, 512), Image.Resampling.BILINEAR)
    pixels = np.asarray(image)
    eligible = valid(pixels)
    values = chroma(pixels)
    medians = []
    for block, keep in zip(np.array_split(values, 8), np.array_split(eligible, 8), strict=True):
        for tile, mask in zip(np.array_split(block, 8, axis=1), np.array_split(keep, 8, axis=1), strict=True):
            candidates = tile[mask]
            count = len(candidates) // 4
            if count >= 32:
                ranks = np.argsort(np.linalg.norm(candidates, axis=1), kind='stable')[:count]
                medians.append(np.median(candidates[ranks], axis=0))
    if not medians:
        return dict(tiles=0, dispersion=None)
    medians = np.stack(medians)
    center = np.median(medians, axis=0)
    dispersion = float(np.median(np.linalg.norm(medians - center, axis=1)))
    return dict(tiles=len(medians), dispersion=dispersion)


def compare(a, b):
    support = valid(a) & valid(b)
    first, second = chroma(a), chroma(b)
    if not support.any():
        raise ValueError('No jointly valid pixels to construct the edit')
    shift = np.median((second - first)[support], axis=0)
    edit, clipped = apply(a, shift)
    common = support & valid(edit)
    if not common.any():
        raise ValueError('No common support after digital edit')
    observed = np.median((second - first)[common], axis=0)
    authored = np.median((chroma(edit) - first)[common], axis=0)
    mismatch = float(np.linalg.norm(observed - authored))
    magnitude = float(np.linalg.norm(observed))
    coverage = float(common.mean())
    qualified = mismatch <= .025 and magnitude >= .08 and coverage >= .10 and clipped <= .02
    images = dict(original=a, relit=b, edited=edit)
    signals = {key: spatial_dispersion(image) for key, image in images.items()}
    measurable = all(s['tiles'] >= 32 for s in signals.values())
    if measurable:
        signal = signals['relit']['dispersion']
        margins = {key: signal - signals[key]['dispersion'] for key in ('original', 'edited')}
    else:
        margins = None
    record = dict(real_shift=observed.tolist(), edited_shift=authored.tolist(), shift_mismatch=mismatch,
                  observed_shift_norm=magnitude, common_coverage=coverage, edit_clipped_fraction=clipped,
                  qualified=qualified, signals=signals, margins=margins,
                  passes=qualified and measurable and min(margins.values()) >= .03)
    return record, images


def run(cache, out, proxy):
    cache.mkdir(exist_ok=True)
    out.mkdir()
    records = {}
    for scene in SCENES:
        paths, archive = intake(scene, cache / scene, proxy)
        images = [decode_image(paths[light]['path']) for light in LIGHTS]
        if any(image.shape != (1000, 1500, 3) for image in images):
            raise ValueError('Expected MIT mip2 dimensions')
        record, panels = compare(*images)
        board = Image.new('RGB', (990, 245), '#202020')
        draw = ImageDraw.Draw(board)
        for index, (name, rgb) in enumerate(panels.items()):
            x = index * 330 + 5
            tile = Image.fromarray(rgb).resize((320, 213), Image.Resampling.LANCZOS)
            draw.text((x, 4), name, fill='white')
            board.paste(tile, (x, 25))
        path = out / f'{scene}-comparison.png'
        board.save(path)
        records[scene] = dict(archive=archive, inputs={k: {name: v for name, v in data.items() if name != 'path'}
                                                      for k, data in paths.items()},
                              edit_sha256=save(out / f'{scene}-edited.png', panels['edited']),
                              comparison_sha256=digest(path), measurement=record)
    write_json(out / 'report.json', records)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
