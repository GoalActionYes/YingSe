"""Measure the same neutral ColorChecker patches in the four pinned captures."""
from argparse import ArgumentParser
from ast import literal_eval
from math import hypot, log2
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


SCENES = ('MIXED_organic', 'WHITE_rich')
LIGHTS = (6, 13)
PATCHES = range(19, 25)


def tagged_number(value):
    if isinstance(value, tuple):
        return value[0] / value[1]
    return float(value)


def measure(folder):
    tags = json.loads((folder / 'oppo_tags.json').read_bytes())
    chart = json.loads((folder / 'oppo_cc_detection.json').read_bytes())
    exif = literal_eval(tags['ExifTag'])
    setting = {key: tagged_number(exif[key]) for key in
               ('ExposureTime', 'ISOSpeedRatings', 'FNumber', 'FocalLength') if key in exif}
    with h5py.File(folder / 'oppo.h5') as handle:
        image = handle['oppo'][:]
    height, width, channels = image.shape
    chart_dimensions = (chart['image_width'], chart['image_height'])
    if channels != 3 or (width, height) != (int(tags['ImageWidth']), int(tags['ImageLength'])):
        raise ValueError('Image and camera tag dimensions differ')
    if chart_dimensions not in ((width, height), (height, width)):
        raise ValueError('Unknown ColorChecker dimension convention')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Expected finite camera RGB in [0, 1]')

    grid = np.linspace(.25, .75, 21)
    u, v = np.meshgrid(grid, grid)
    weights = np.stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v), axis=-1)
    centers = []
    patches = {}
    for number in range(1, 25):
        corners = np.asarray(chart[f'patch_{number}']['corners'])
        if corners.shape != (4, 2) or np.any(corners < 0) or np.any(corners >= [width, height]):
            raise ValueError('ColorChecker corners lie outside the image')
        centers.append(corners.mean(axis=0))
        if number in PATCHES:
            x, y = np.rint(weights @ corners).astype(int).transpose(2, 0, 1)
            values = image[y, x]
            median = np.median(values, axis=(0, 1)).astype(np.float64)
            total = float(median.sum())
            patches[number] = dict(
                rgb_median=median.tolist(),
                normalized_rgb=(median / total).tolist() if total > 0 else None,
                rgb_endpoint_fraction=float(np.any(values >= 1, axis=-1).mean()),
                sensor_clip_fraction=None,
            )
    info = dict(dimensions=[width, height], chart_tag_dimensions=chart_dimensions, model=tags['Model'], orientation=tags['Orientation'],
                chart_type=chart['chart_type'], dng_white_level=tags.get('WhiteLevel'),
                settings=setting, neutral_patches=patches,
                member_sha256=json.loads((folder / 'members.json').read_bytes()))
    return info, np.array(centers), image


def exposure(info):
    tags = info['settings']
    keys = ('ExposureTime', 'ISOSpeedRatings', 'FNumber')
    if not all(key in tags for key in keys):
        return None
    return log2(tags['ExposureTime'] * tags['ISOSpeedRatings'] / tags['FNumber'] ** 2)


def compare(first, second, centers):
    geometry = float(np.linalg.norm(centers[0] - centers[1], axis=1).max() / hypot(*first['dimensions']))
    a, b = exposure(first), exposure(second)
    ev_difference = abs(a - b) if a is not None and b is not None else None
    distances = {}
    usable = []
    for number in PATCHES:
        p, q = (entry['neutral_patches'][number] for entry in (first, second))
        if p['normalized_rgb'] is not None and q['normalized_rgb'] is not None:
            distance = float(np.linalg.norm(np.subtract(p['normalized_rgb'], q['normalized_rgb'])))
            distances[number] = distance
            if max(p['rgb_endpoint_fraction'], q['rgb_endpoint_fraction']) < .01 and distance >= .015:
                usable.append(number)
    match = all(first[key] == second[key] for key in ('model', 'orientation', 'dimensions', 'chart_type'))
    match = match and first['settings'].get('FocalLength') == second['settings'].get('FocalLength')
    return dict(camera_layout_matches=match, max_patch_center_shift_diagonal_fraction=geometry,
                exposure_difference_ev=ev_difference, gray_patch_rgb_distances=distances,
                usable_rgb_patches=usable, geometry_pass=match and geometry <= .01,
                exposure_pass=ev_difference is not None and ev_difference <= .1,
                rgb_diagnostic_pass=len(usable) >= 3,
                sensor_clip_pass=None)


def run(source, report, preview, *, batch=False):
    results = {}
    board = Image.new('RGB', (1000, 830), '#202020')
    draw = ImageDraw.Draw(board)
    draw.text((12, 12), 'Oppo camera RGB | fixed gamma 1/2.2 | no white balance', fill='white')
    scenes = ('WHITE_green', 'BLUE_red') if batch else SCENES
    lights = (1, 8) if batch else LIGHTS
    for row, scene in enumerate(scenes):
        captures = {}
        centers = []
        for column, light in enumerate(lights):
            info, points, raw = measure(source / f'{scene}-light{light}')
            captures[light] = info
            centers.append(points)
            pixels = np.rint(np.power(raw[::4, ::4], 1/2.2) * 255).astype(np.uint8)
            tile = Image.fromarray(pixels).resize((480, 360), Image.Resampling.LANCZOS)
            x, y = 12 + column * 496, 40 + row * 393
            draw.text((x, y), f'{scene} | light {light}', fill='white')
            board.paste(tile, (x, y + 20))
            for number in PATCHES:
                patch_x, patch_y = points[number - 1]
                dot_x = x + patch_x * 480 / info['dimensions'][0]
                dot_y = y + 20 + patch_y * 360 / info['dimensions'][1]
                draw.ellipse((dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3), outline='white')
        results[scene] = dict(captures=captures, comparison=compare(*captures.values(), centers))
    with report.open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(results, indent=2, allow_nan=False) + '\n')
    board.save(preview)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for argument in ('source', 'report', 'preview'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--batch', action='store_true', help='Audit the exposure-screened captures')
    run(**vars(parser.parse_args()))
