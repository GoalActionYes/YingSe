"""Measure transfer of paired archival chroma adjustments at frozen target luma."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_sparks_displays import PATCHES, neutral, read_run
from probe_sparks_archival_grade import FRAMES, sha, write_json
from probe_sparks_shared_tone import LUMA

PROTOCOL = Path(__file__).with_name('sparks-shared-chroma-protocol.md')
ROLES = ('anchor', 'output', 'constant', 'cross', 'within', 'gam')
LABELS = ('Original color / fixed Y', 'YingSe 13', 'Constant chroma', 'Other-shot chroma', 'Same-shot chroma', 'Author GAM')


def chroma(rgb, light):
    return rgb[..., [0, 2]] - light[..., None]


def fit_curve(original, author, selected):
    light = original @ LUMA
    difference = chroma(author, author @ LUMA) - chroma(original, light)
    bands = np.floor(light / 16).astype(int)
    nodes = {}
    for band in range(16):
        use = selected & (bands == band)
        count = int(use.sum())
        if count >= 64:
            nodes[band] = dict(light=float(np.median(light[use])),
                               chroma=np.median(difference[use], axis=0).tolist(), count=count)
    return nodes, np.median(difference[selected], axis=0).tolist()


def predict(nodes, light):
    shift = np.zeros((*light.shape, 2))
    support = np.zeros(light.shape, dtype=bool)
    for key in nodes:
        if key + 1 not in nodes:
            continue
        left, right = (nodes[k]['light'] for k in (key, key + 1))
        take = (light >= left) & (light <= right)
        for axis in range(2):
            ends = [nodes[k]['chroma'][axis] for k in (key, key + 1)]
            shift[take, axis] = np.interp(light[take], (left, right), ends)
        support |= take
    return shift, support


def render(color, light):
    red, blue = np.moveaxis(color, -1, 0)
    green = -(LUMA[0] * red + LUMA[2] * blue) / LUMA[1]
    component = np.stack((red, green, blue), axis=-1)
    pole = light[..., None]
    upper = np.divide(255 - pole, component, out=np.ones_like(component), where=component > 0)
    lower = np.divide(-pole, component, out=np.ones_like(component), where=component < 0)
    scale = np.minimum(1., np.minimum(upper, lower).min(axis=-1))
    rgb = np.rint(np.clip(pole + component * scale[..., None], 0, 255)).astype(np.uint8)
    return rgb, float((scale < 1. - 1e-10).mean())


def metric(images, masks):
    author = images['gam']
    truth = chroma(author, author @ LUMA)
    result = {}
    for role in ROLES[:-1]:
        image = images[role]
        error = np.abs(chroma(image, image @ LUMA) - truth)
        result[role] = {name: float(error[mask].mean()) if mask.any() else None for name, mask in masks.items()}
    return result


def panel(frames, path):
    width, height = 384, 203
    sheet = Image.new('RGB', ((width + 8) * len(ROLES) + 8, 2 * (height + 34) + 8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for row, frame in enumerate(FRAMES):
        for column, (role, label) in enumerate(zip(ROLES, LABELS)):
            x, y = 8 + column * (width + 8), 8 + row * (height + 34)
            draw.text((x, y), f'{frame}  {label}', font=font, fill='#f1f3f6')
            thumb = Image.fromarray(frames[frame][role]).resize((width, height), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y + 24))
    sheet.save(path, quality=92, subsampling=0)


def run(prepared_dir, baseline_dir, evidence, out):
    start = time.monotonic()
    frozen = json.loads(evidence.read_bytes())
    prepared, baseline, frames, identity = read_run(prepared_dir, baseline_dir)
    if prepared != frozen['prepared'] or baseline != frozen['aces_render'] or baseline['strength'] != .8:
        raise ValueError('Unexpected ACES development inputs')
    y, x = np.indices(frames[FRAMES[0]]['nam'].shape[:2])
    split = (y // 16 + x // 16) % 2
    models = {k: [fit_curve(im['nam'], im['gam'], split != fold) for fold in (0, 1)]
              for k, im in frames.items()}
    out.mkdir()
    cases = {}
    for frame, donor in (FRAMES, FRAMES[::-1]):
        images = frames[frame]
        original = images['nam']
        old_light = original @ LUMA
        fixed_light = images['output'] @ LUMA
        base = chroma(original, old_light)
        deltas = {key: np.zeros_like(base) for key in ('anchor', 'constant', 'cross', 'within')}
        support = {key: np.zeros(split.shape, dtype=bool) for key in ('cross', 'within')}
        for fold in (0, 1):
            held = split == fold
            for role, model in (('cross', donor), ('within', frame)):
                shift, covered = predict(models[model][fold][0], old_light)
                deltas[role][held] = shift[held]
                support[role] |= held & covered
            deltas['constant'][held & support['cross'] & support['within']] = models[donor][fold][1]

        common = support['cross'] & support['within']
        name, x1, y1, x2, y2 = PATCHES[frame]
        roi = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        masks = dict(full=np.ones_like(common), common=common, common_neutral=common & neutral(original), **{name: roi})
        records = {}
        for role, delta in deltas.items():
            images[role], compressed = render(base + .8 * delta, fixed_light)
            path = out / f'{frame}-{role}.png'
            Image.fromarray(images[role]).save(path)
            records[role] = dict(sha256=sha(path), compressed=compressed,
                                 max_luma_error=float(np.abs(images[role] @ LUMA - fixed_light).max()))

        scores = metric(images, masks)
        cross = scores['cross']
        checks = dict(coverage=common.mean() >= .50, neutral_coverage=masks['common_neutral'].sum() >= 256,
                      luma_preserved=all(row['max_luma_error'] <= .5 for row in records.values()))
        if common.any():
            checks.update(reduction=cross['common'] <= .8 * scores['anchor']['common'],
                          constant_margin=cross['common'] <= scores['constant']['common'] - 1.,
                          within_margin=cross['common'] <= scores['within']['common'] + 2.)
        for region in ('common_neutral', name):
            checks[region] = cross[region] is not None and cross[region] <= min(
                scores['anchor'][region], scores['output'][region]) + 2.
        cases[frame] = dict(donor=donor, pixels={k:int(mask.sum()) for k, mask in masks.items()},
                            scores=scores, outputs=records, checks={k:bool(v) for k,v in checks.items()},
                            passed=all(checks.values()))

    panel(frames, out / 'comparison.jpg')
    if time.monotonic() - start > 120:
        raise RuntimeError('Run exceeded the compute budget')
    result = dict(cases=cases, models=models, passed=all(c['passed'] for c in cases.values()),
                  aces_parent_sha256=sha(evidence), baseline_sha256=identity,
                  source_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                  comparison_sha256=sha(out / 'comparison.jpg'))
    write_json(out / 'results.json', result)
    for frame, row in cases.items():
        print(frame, row['checks'], row['scores'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('prepared_dir', 'baseline_dir', 'evidence', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
