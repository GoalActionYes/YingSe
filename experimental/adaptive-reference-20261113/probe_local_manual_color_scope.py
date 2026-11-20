"""Compare scoped and global use of the existing temperature and tint controls."""
from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from audit_bg_real_skin_regions import BASE, LUMA, PROTOCOL, ROOT, digest, read
from probe_beyond_rgb_transfer import save, write_json
from reference_color_app.adjustments import transform


SOURCE = '83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9'
REGIONS = '2e0fe1e6b0b47e3579263bbd7cee626713955fee6c3cde71ef8323784de702b1'
SKIP = {'雪地', '白球服', '投影幕中的脸'}


def patches():
    if digest(PROTOCOL) != REGIONS:
        raise ValueError('Changed original region protocol')
    pattern = r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$'
    result = {}
    for target, reference, label, *values in re.findall(pattern, PROTOCOL.read_text(encoding='utf-8'), re.M):
        result.setdefault(f'{target}-{reference}', {})[label] = tuple(map(int, values))
    return result


def adjust(rgb, settings):
    x = rgb.astype(np.float32) / 255
    return np.rint(transform(x, {'basic': settings}) * 255).astype(np.uint8)


def measure(delta):
    value = delta.astype(np.float64)
    luma = value @ LUMA
    return dict(chroma=float(np.abs(value - luma[..., None]).mean()),
                luma=float(np.abs(luma).mean()), rgb=float(np.abs(value).mean()))


def tune(given, anchor, samples):
    def loss(item):
        temperature, tint = item
        params = dict(temperature=temperature, tint=tint)
        prediction = adjust(given[samples], params)
        return (measure(prediction.astype(np.int16) - anchor[samples])['chroma'],
                abs(temperature) + abs(tint), temperature, tint)

    temperature, tint = min(product(range(-20, 21, 5), repeat=2), key=loss)
    return dict(temperature=temperature, tint=tint)


def edit_pair(active, anchor, bounds, all_patches):
    left, top, right, bottom = bounds
    yy, xx = np.indices(active.shape[:2])
    distance = np.maximum.reduce([left-xx, xx-(right-1), top-yy, yy-(bottom-1), np.zeros_like(yy)])
    core = distance == 0
    train = core & ((xx + yy) % 2 == 0)
    held = core & ~train
    params = tune(active, anchor, train)
    whole = adjust(active, params)
    weight = np.clip(1 - distance / 8, 0, 1)
    localized = np.rint(active + weight[..., None] * (whole.astype(np.int16) - active)).astype(np.uint8)
    np.testing.assert_array_equal(localized[weight == 0], active[weight == 0])
    initial = measure(active[held] - anchor[held])

    scores = {}
    for name, rgb in (('global', whole), ('scoped', localized)):
        change = rgb.astype(np.int16) - active
        checks = {key: measure(change[t:b, l:r])
                  for key, (l, t, r, b) in all_patches.items() if (l, t, r, b) != bounds}
        after = measure(rgb[held].astype(np.int16) - anchor[held])
        scores[name] = dict(held=after, surroundings=measure(change[(weight > 0) & ~core]),
                            elsewhere=measure(change[weight == 0]), other_patches=checks,
                            passes=after['chroma'] <= min(1.5, .85 * initial['chroma']))
    return (whole, localized), dict(parameters=params, held_pixels=int(held.sum()), before=initial, variants=scores,
                                   local_weight_pixels=int((weight > 0).sum()))


def run(out):
    file = ROOT / 'docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json'
    if digest(file) != SOURCE:
        raise ValueError('Changed input pairs')
    known = patches()
    out.mkdir()
    results = {}
    for key, row in json.loads(file.read_bytes())['cases'].items():
        pair = row['target_pair']
        anchor, active = [read(BASE/'bg-real-pixel-pilot'/f'{key}-{tag}.png', row['outputs'][tag])
                          for tag in ('C1','F1')]
        for index, (label, area) in enumerate(known[pair].items()):
            if label in SKIP:
                continue
            images, report = edit_pair(active, anchor, area, known[pair])
            name = f'{pair}-{index}'
            report['label'] = label
            report['sha256'] = {kind: save(out / f'{name}-{kind}.png', image)
                                for kind, image in zip(('global','scoped'), images, strict=True)}
            board = Image.new('RGB', (1056, 284), '#242630')
            draw = ImageDraw.Draw(board)
            for index, (kind, image) in enumerate(zip(('Anchor','F1 input','Global','Scoped'),
                                                     (anchor.astype(np.uint8),active.astype(np.uint8),*images),
                                                     strict=True)):
                x = 4 + 264 * index
                tile = Image.fromarray(image)
                board.paste(tile, (x, 22))
                draw.text((x, 4), kind, fill='white')
                l, t, r, b = area
                draw.rectangle((x + l - 1, 21 + t, x + r, 22 + b), outline='#ffe020')
            path = out / f'{name}-comparison.png'
            board.save(path)
            results[name] = dict(report, panel_sha256=digest(path))
    if len(results) != 6:
        raise ValueError('Changed number of measured targets')
    write_json(out/'report.json', dict(source_sha256=SOURCE, regions_sha256=REGIONS, cases=results))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
