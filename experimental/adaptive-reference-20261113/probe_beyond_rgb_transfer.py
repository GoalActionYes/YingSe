"""Render and score a crossed reference-light intervention with frozen pipelines."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCENES = ('MIXED_organic', 'WHITE_rich')
LIGHTS = (6, 13)
PATCHES = tuple(range(19, 24))
CAPTURE_AUDIT = '21560ea26bfcfa90adcb5dd1bcdad326ea7f3ec1e9f889274c1a16ee9435fd75'
BATCH_CAPTURE_AUDIT = '0417b3008047042e1e9684da7c55bcea62779b4dd6757b2fb071e4dca2cdf873'
STRENGTH = .8
LUMA = np.array([.2126, .7152, .0722])
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -.4985314],
                        [-.9692660, 1.8760108, .0415560],
                        [.0556434, -.2040259, 1.0572252]])


def digest(path):
    with path.open('rb') as handle:
        return sha256(handle.read()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def prepare(source, evidence, out, headroom=False, *, batch=False):
    import h5py

    audit_bytes = evidence.read_bytes().replace(b'\r\n', b'\n')
    audit_sha = BATCH_CAPTURE_AUDIT if batch else CAPTURE_AUDIT
    if sha256(audit_bytes).hexdigest() != audit_sha:
        raise ValueError('Changed source audit')
    selection = json.loads(audit_bytes)
    out.mkdir()
    records = {}
    matrices = []
    linear_images = {}
    scenes = ('WHITE_green', 'BLUE_red') if batch else SCENES
    lights = (1, 8) if batch else LIGHTS
    for scene in scenes:
        for light in lights:
            key = f'{scene}-{light}'
            folder = source / f'{scene}-light{light}'
            audit = selection[scene]['captures'][str(light)]
            for name, expected in audit['member_sha256'].items():
                if digest(folder / name) != expected:
                    raise ValueError(f'Changed capture: {key}/{name}')
            tags = json.loads((folder / 'oppo_tags.json').read_bytes())
            if int(tags['CalibrationIlluminant1']) != 21:
                raise ValueError('Expected fixed D65 calibration')
            matrix = np.fromstring(tags['ColorMatrix1'].strip('[]'), sep=' ').reshape((3, 3))
            matrices.append(matrix)
            with h5py.File(folder / 'oppo.h5') as handle:
                rgb = handle['oppo'][1::4, 1::4]
            linear = rgb @ (XYZ_TO_SRGB @ np.linalg.inv(matrix)).T
            linear_images[key] = linear
            chart = json.loads((folder / 'oppo_cc_detection.json').read_bytes())
            coordinates = {n: ((np.asarray(chart[f'patch_{n}']['corners']) - 1) / 4).tolist()
                           for n in PATCHES}
            records[key] = dict(patches=coordinates)
    bounds = [float(min(linear.min() for linear in linear_images.values())),
              float(max(linear.max() for linear in linear_images.values()))]
    for key, linear in linear_images.items():
        gamut = np.any((linear < 0) | (linear > 1), axis=-1)
        if headroom:
            linear = .02 + .96 * (linear - bounds[0]) / (bounds[1] - bounds[0])
        linear = np.clip(linear, 0, 1)
        encoded = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1/2.4) - .055)
        image = np.rint(encoded * 255).astype(np.uint8)
        records[key].update(image_sha256=save(out / f'{key}.png', image),
                            gamut_limited_fraction=float(gamut.mean()))
    if any(not np.array_equal(matrix, matrices[0]) for matrix in matrices):
        raise ValueError('The captures do not share one D65 calibration')
    write_json(out / 'inputs.json', dict(source_sha256=audit_sha, records=records,
                                          headroom=headroom, linear_bounds=bounds))


def patch_measurement(image, corners):
    u, v = np.meshgrid(np.linspace(.25, .75, 21), np.linspace(.25, .75, 21))
    weights = np.stack(((1-u)*(1-v), u*(1-v), u*v, (1-u)*v), axis=-1)
    x, y = np.rint(weights @ corners).astype(int).transpose(2, 0, 1)
    samples = image[y, x]
    median = np.median(samples, axis=(0, 1))
    luma = float(median @ LUMA)
    return dict(rgb=median.tolist(), luma=luma, chroma=(median - luma).tolist(),
                endpoint_fraction=float(np.any((samples == 0) | (samples == 255), axis=-1).mean()))


def projection(guide_pair, output_pair):
    guide = np.subtract(*(patch['chroma'] for patch in guide_pair[::-1]))
    output = np.subtract(*(patch['chroma'] for patch in output_pair[::-1]))
    span = float(np.linalg.norm(guide))
    gain = float(output @ guide / (span ** 2)) if span else None
    return dict(reference_span=span, output_span=float(np.linalg.norm(output)), projection_ratio=gain,
                orthogonal=float(np.linalg.norm(output - gain * guide)) if gain is not None else None,
                measurable=span >= 18, carried=span >= 18 and gain >= .20)


def run(prepared, out):
    import torch

    from reference_color_app.lut import apply_mapping
    from reference_color_app.pipeline import algorithm_fingerprint, analyze
    from spatial_render import SpatialEngine, finish_spatial

    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    inputs = prepared / 'inputs.json'
    manifest = json.loads(inputs.read_bytes())
    if manifest['source_sha256'] != CAPTURE_AUDIT:
        raise ValueError('Changed prepared source')
    records = manifest['records']
    paths = {key: prepared / f'{key}.png' for key in records}
    images = {}
    for key, path in paths.items():
        if digest(path) != records[key]['image_sha256']:
            raise ValueError(f'Changed prepared image: {key}')
        with Image.open(path) as image:
            images[key] = np.array(image)
    out.mkdir()
    cases = {}
    for target_scene, guide_scene in permutations(SCENES):
        for target_light in LIGHTS:
            target_key = f'{target_scene}-{target_light}'
            ref_keys = tuple(f'{guide_scene}-{light}' for light in LIGHTS)
            target = images[target_key]
            artifacts = {}
            rendered = {'stable': [], 'spatial13': []}
            for guide_key in ref_keys:
                guide = images[guide_key]
                mapping = analyze(paths[target_key], paths[guide_key])
                results = {
                    'stable': apply_mapping(target, mapping, STRENGTH),
                    'spatial13': finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH),
                }
                for key, result in results.items():
                    rendered[key].append(result)
                    name = f'{target_key}-from-{guide_key}-{key}.png'
                    artifacts[name] = save(out / name, result)
            summaries = {}
            for key, frames in rendered.items():
                measurements = {}
                for patch in PATCHES:
                    corners = records[target_key]['patches'][str(patch)]
                    guide_pair = [patch_measurement(images[ref], records[ref]['patches'][str(patch)])
                                  for ref in ref_keys]
                    graded_pair = [patch_measurement(frame, corners) for frame in frames]
                    original = patch_measurement(target, corners)
                    measurements[patch] = dict(target=original, reference=guide_pair, output=graded_pair,
                                               **projection(guide_pair, graded_pair))
                changes = [float(np.abs(frame.astype(np.int16) - target).mean()) for frame in frames]
                summaries[key] = dict(patches=measurements, mean_rgb8_change_from_target=changes,
                                      mean_rgb8_change_between_outputs=float(
                                          np.abs(frames[1].astype(np.int16) - frames[0]).mean()))
            case = f'{target_key}-from-{guide_scene}'
            board = Image.new('RGB', (1036, 404), '#222730')
            draw = ImageDraw.Draw(board)
            panels = [('Target', target), ('Reference 6', images[ref_keys[0]]),
                      ('Stable / reference 6', rendered['stable'][0]), ('Spatial / reference 6', rendered['spatial13'][0]),
                      ('Target', target), ('Reference 13', images[ref_keys[1]]),
                      ('Stable / reference 13', rendered['stable'][1]), ('Spatial / reference 13', rendered['spatial13'][1])]
            for index, (label, image) in enumerate(panels):
                x, y = (index % 4) * 259 + 4, (index // 4) * 202
                tile = Image.fromarray(image).resize((248, 186), Image.Resampling.LANCZOS)
                draw.text((x, y + 1), label, fill='white')
                board.paste(tile, (x, y + 15))
            path = out / f'{case}-comparison.png'
            board.save(path)
            cases[case] = dict(outputs_sha256=artifacts, panel_sha256=digest(path), pipelines=summaries)
    write_json(out / 'report.json', dict(prepared_sha256=digest(inputs),
               checkpoint_sha256=engine.checkpoint_sha, stable_fingerprint=algorithm_fingerprint(),
               strength=STRENGTH, headroom=manifest['headroom'],
               linear_bounds=manifest['linear_bounds'], inputs=records, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, arguments in (('prepare', ('source', 'evidence', 'out')), ('render', ('prepared', 'out'))):
        entry = commands.add_parser(command)
        for argument in arguments:
            entry.add_argument(argument, type=Path)
        if command == 'prepare':
            entry.add_argument('--headroom', action='store_true')
            entry.add_argument('--batch', action='store_true')
    args = vars(parser.parse_args())
    {'prepare': prepare, 'render': run}[args.pop('command')](**args)
