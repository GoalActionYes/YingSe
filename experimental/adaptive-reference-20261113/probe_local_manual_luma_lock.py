"""Compare identical scoped manual corrections with and without an RGB8 luma lock."""
from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_bg_real_skin_regions import BASE, LUMA, ROOT, digest
from probe_beyond_rgb_transfer import save, write_json
from probe_local_manual_color_scope import SOURCE, adjust, measure, tune


EXPOSURE = '7e5137f4e559daf39e4f4f09a7500637beb897713257c4d7101de20a0d9dbf97'
EDITS = '5ab0a185c1e4ed6010e196597e5e44875ce9974ba5fa8d7ae742854b07a3aa28'
RENDER = 'b16bd8e38e07f9eb7ea70762079cb6f2780526735c6a3957e7be749fde0c677c'


def load(path, expected):
    if digest(path) != expected:
        raise ValueError('Changed source manifest')
    return json.loads(path.read_bytes())


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError('Changed source image')
    with Image.open(path) as image:
        return np.array(image)


def montage(path, rows):
    board = Image.new('RGB', (640, 514), '#242630')
    draw = ImageDraw.Draw(board)
    for row, panels in enumerate(rows):
        for column, (label, frame) in enumerate(panels):
            x, y = column * 320 + 5, row * 255 + 4
            tile = Image.fromarray(frame)
            tile.thumbnail((310, 230), Image.Resampling.LANCZOS)
            draw.text((x, y), label, fill='white')
            board.paste(tile, (x, y + 17))
    board.save(path)
    return digest(path)

def read_cases():
    source = BASE / 'dast-exposure-prepared-v1'
    edited = BASE / 'dast-local-edit-prepared-v1'
    rendered = BASE / 'dast-local-edit-cross-v1'
    photos = load(source / 'inputs.json', EXPOSURE)
    controls = load(edited / 'edits.json', EDITS)
    output = load(rendered / 'results.json', RENDER)
    groups = {str(group['group']): group for group in photos['groups']}
    for row in output['groups']:
        reference = str(row['group'])
        target = str(groups[reference]['target_group'])
        images = row['output_sha256']
        original = verified(source / target / 'nominal.png', groups[target]['frames']['nominal'])
        face = verified(source / target / 'nominal-mask.png', groups[target]['target_mask_sha256']) > 0
        truth = verified(edited / target / 'truth.png', controls['groups'][target]['nominal']['truth_sha256'])
        active = verified(rendered / reference / 'nominal-on.png', images['nominal-on'])
        off = verified(rendered / reference / 'nominal-off.png', images['nominal-off'])
        yield f'face-{reference}', face, active, truth, original, off

    records = load(ROOT / 'docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json', SOURCE)
    name, row = next((key, value) for key, value in records['cases'].items() if value['target_pair'] == '9483-19432')
    anchor, active = (verified(BASE / 'bg-real-pixel-pilot' / f'{name}-{state}.png', row['outputs'][state])
                      for state in ('C1', 'F1'))
    wall = np.zeros(active.shape[:2], dtype=bool)
    wall[:15, :232] = True
    wall[:66, :28] = True
    yield 'wall-9483', wall, active, anchor, None, None


def locked_adjustment(base, rgb):
    rgb = rgb.astype(float)
    pole = (base @ LUMA)[..., None]
    color = rgb - (rgb @ LUMA)[..., None]
    upper = np.divide(255 - pole, color, out=np.ones_like(color), where=color > 0)
    lower = np.divide(-pole, color, out=np.ones_like(color), where=color < 0)
    scale = np.minimum(1., np.minimum(upper, lower).min(axis=-1))
    output = np.clip(pole + scale[..., None] * color, 0, 255)
    return output, scale


def apply(scope, active, expected, original, off):
    if scope.shape != active.shape[:2] or active.shape != expected.shape:
        raise ValueError('Scope resolution differs from active image')
    padded = np.pad(scope.astype(np.uint8), 1)
    depth = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    weight = np.clip(depth / 8, 0, 1)
    yy, xx = np.indices(scope.shape)
    train = (weight == 1) & ((xx // 16 + yy // 16) % 2 == 0)
    held = (weight == 1) & ~train
    edge = scope & (weight < 1)
    if min(int(train.sum()), int(held.sum()), int(edge.sum())) < 128:
        raise ValueError('Insufficient fully weighted or edge support')
    match = tune(active, expected.astype(np.int16), train)
    adjusted = adjust(active, match)
    color_only, scale = locked_adjustment(active, adjusted)
    before = measure(active[held].astype(np.int16) - expected[held])
    outputs, scores = {}, {}
    for arm, raw in (('manual', adjusted), ('locked', color_only)):
        delta = raw.astype(float) - active
        output = np.rint(active + weight[..., None] * delta).astype(np.uint8)
        np.testing.assert_array_equal(output[~scope], active[~scope])
        difference = output.astype(np.int16) - active
        evaluation = measure(output[held].astype(np.int16) - expected[held])
        drift = float(np.abs(difference @ LUMA).max())
        measured = dict(held=evaluation, edge=measure(difference[edge]), luma_drift=drift,
                        outside=measure(difference[~scope]),
                        chroma_pass=evaluation['chroma'] <= min(1.5, .85 * before['chroma']))
        if original is not None:
            truth = expected[held].astype(float) - original[held]
            response = output[held].astype(float) - off[held]
            measured['style_projection'] = float(np.sum(response * truth) / np.square(truth).sum())
        scores[arm] = measured
        outputs[arm] = output
    if scores['locked']['luma_drift'] > .5001:
        raise ValueError('Luma lock drift exceeded RGB8 quantization')
    return outputs, dict(before=before, held_pixels=int(held.sum()), edge_pixels=int(edge.sum()),
                         parameters=match, scores=scores, limited_fraction=float((scale[scope] < 1).mean()))


def run(out):
    results = {}
    out.mkdir()
    for key, scope, base, expected, original, off in read_cases():
        outputs, record = apply(scope, base, expected, original, off)
        figures = {arm: save(out / f'{key}-{arm}.png', image) for arm, image in outputs.items()}
        panel = montage(out / f'{key}-comparison.png',
                        [[('Grade before repair', base), ('Anchor', expected)],
                         [(f'Scoped {arm}', image) for arm, image in outputs.items()]])
        results[key] = dict(record, outputs_sha256=figures, panel_sha256=panel)
    write_json(out/'report.json', results)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
