"""Render an offline cross-shot luminance adjustment on two fixed Sparks pairs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_sparks_displays import PATCHES, neutral, read_run
from probe_sparks_archival_grade import FRAMES, sha, write_json

LUMA = np.array([.2126, .7152, .0722])
PROTOCOL = Path(__file__).with_name('sparks-shared-tone-protocol.md')
ROLES = ('nam', 'output', 'constant', 'cross', 'within', 'gam')
LABELS = ('Original NAM', 'YingSe 13', 'Offset control', 'Other-shot curve', 'Same-shot curve', 'Author GAM')


def fit_curve(original, author, train):
    light = original @ LUMA
    difference = author @ LUMA - light
    nodes = {}
    for index in range(16):
        selected = train & (light >= index * 16) & (light < 16 * (index + 1))
        count = int(selected.sum())
        if count >= 64:
            nodes[index] = (float(np.median(light[selected])), float(np.median(difference[selected])), count)
    return nodes, float(np.median(difference[train]))


def predict(nodes, light):
    delta = np.zeros(light.shape)
    supported = np.zeros(light.shape, dtype=bool)
    for index, (left, low, _) in nodes.items():
        if index + 1 in nodes:
            right, high, _ = nodes[index + 1]
            selected = (light >= left) & (light <= right)
            delta[selected] = low + (light[selected] - left) * (high - low) / (right - left)
            supported |= selected
    return delta, supported


def apply_difference(original, shift):
    rgb = original.astype(np.int16)
    rounded = np.rint(.8 * shift).astype(np.int16)
    limited = np.clip(rounded, -rgb.min(-1), 255 - rgb.max(-1))
    return (rgb + limited[..., None]).astype(np.uint8)


def evaluate(im, mask):
    region = {key: value[mask].astype(np.float64) for key, value in im.items()}
    author = region['gam']
    return {
        role: dict(rgb_mae=float(np.abs(region[role] - author).mean()),
                   luma_mae=float(np.abs((region[role] - author) @ LUMA).mean()))
        for role in ROLES if role != 'gam'
    } if mask.any() else None


def compare_panel(frames, path):
    size = (384, 203)
    width, height = size
    sheet = Image.new('RGB', ((width + 8) * len(ROLES) + 8, (height + 34) * 2 + 8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for row, frame in enumerate(FRAMES):
        for column, (role, label) in enumerate(zip(ROLES, LABELS)):
            x, y = 8 + column * (width + 8), 8 + row * (height + 34)
            draw.text((x, y), f'{frame}  {label}', font=font, fill='#f1f3f6')
            image = Image.fromarray(frames[frame][role]).resize(size, Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y + 24))
    sheet.save(path, quality=92, subsampling=0)


def run(prepared_dir, baseline_dir, evidence, out):
    start = time.monotonic()
    parent = json.loads(evidence.read_bytes())
    prepared, baseline, frames, identity = read_run(prepared_dir, baseline_dir)
    if prepared != parent['prepared'] or baseline != parent['aces_render'] or baseline['strength'] != .8:
        raise ValueError('Unexpected ACES development inputs')
    height, width = frames[FRAMES[0]]['nam'].shape[:2]
    y, x = np.indices((height, width))
    split = ((y // 16 + x // 16) % 2).astype(bool)
    models = {frame: [fit_curve(im['nam'], im['gam'], split != test) for test in (False, True)]
              for frame, im in frames.items()}
    out.mkdir()
    cases = {}
    for frame, donor in (FRAMES, FRAMES[::-1]):
        images = frames[frame]
        original = images['nam']
        light = original @ LUMA
        cross = np.zeros_like(light)
        within = np.zeros_like(light)
        constant = np.zeros_like(light)
        common = np.zeros_like(split)
        supports = {role: np.zeros_like(split) for role in ('cross', 'within')}
        for fold, (cross_fit, own_fit) in enumerate(zip(models[donor], models[frame])):
            test = split == fold
            for role, fitted, changes in (('cross', cross_fit, cross), ('within', own_fit, within)):
                delta, covered = predict(fitted[0], light)
                changes[test] = delta[test]
                supports[role][test] = covered[test]
            supported = test & supports['cross'] & supports['within']
            common |= supported
            constant[supported] = cross_fit[1]
        output_hashes = {}
        for role, delta in (('cross', cross), ('within', within), ('constant', constant)):
            images[role] = apply_difference(original, delta)
            path = out / f'{frame}-{role}.png'
            Image.fromarray(images[role]).save(path)
            output_hashes[role] = sha(path)

        name, x1, y1, x2, y2 = PATCHES[frame]
        scope = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        masks = dict(full=np.ones_like(common), common=common, common_neutral=common & neutral(original))
        masks[name] = scope
        metrics = {key: evaluate(images, mask) for key, mask in masks.items()}
        summary = metrics['common']
        checks = dict(common_coverage=common.mean() >= .5, neutral_coverage=masks['common_neutral'].sum() >= 256,
                      luma_reduction=summary is not None and summary['cross']['luma_mae'] <= .8 * summary['nam']['luma_mae'],
                      within_margin=summary is not None and summary['cross']['luma_mae'] <= summary['within']['luma_mae'] + 2,
                      constant_margin=summary is not None and summary['cross']['luma_mae'] <= summary['constant']['luma_mae'] - 1)
        cases[frame] = dict(donor=donor, coverage={k: float(v.mean()) for k, v in supports.items()},
                            pixels={key: int(mask.sum()) for key, mask in masks.items()},
                            patch_common_pixels=int((scope & common).sum()), metrics=metrics,
                            checks={k: bool(v) for k, v in checks.items()}, passed=all(checks.values()),
                            output_sha256=output_hashes)
    compare_panel(frames, out / 'comparison.jpg')
    if time.monotonic() - start > 120:
        raise RuntimeError('Run exceeded the compute budget')
    write_json(out / 'results.json', dict(cases=cases, models=models, passed=all(r['passed'] for r in cases.values()),
                                          aces_parent_sha256=sha(evidence), baseline_sha256=identity,
                                          source_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                                          comparison_sha256=sha(out / 'comparison.jpg')))
    for frame, row in cases.items():
        print(frame, row['pixels'], row['checks'], row['metrics']['common'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('prepared_dir', 'baseline_dir', 'evidence', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
