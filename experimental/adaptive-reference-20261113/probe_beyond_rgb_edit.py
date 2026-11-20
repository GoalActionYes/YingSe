"""Isolate the response to a known desaturation across measured reference lighting."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from controlled_styles import stylize
from probe_arri_shared_look import metrics
from probe_beyond_rgb_transfer import (
    LIGHTS, PATCHES, SCENES, STRENGTH, digest, patch_measurement, projection, save, write_json,
)
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial


BASELINE = 'ad2b10f42558ec951c430ae823438444e783b568d1c05ec16e13c099d03ffa1e'
EDIT = 'global_desaturate'


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Changed input: {path.name}')
    with Image.open(path) as image:
        return np.array(image)


def make_masks(target, patches):
    masks = {'whole': np.ones(target.shape[:2], dtype=bool)}
    for patch in PATCHES:
        polygon = np.asarray(patches[str(patch)])
        corners = polygon.mean(axis=0) + .5 * (polygon - polygon.mean(axis=0))
        raster = Image.new('1', (target.shape[1], target.shape[0]))
        ImageDraw.Draw(raster).polygon([tuple(point) for point in corners], fill=1)
        masks[str(patch)] = np.asarray(raster)
    return masks


def montage(path, rows):
    board = Image.new('RGB', (1150, 800), '#242630')
    draw = ImageDraw.Draw(board)
    for row, items in enumerate(rows):
        for column, (label, image) in enumerate(items):
            x, y = 4 + column * 230, 4 + row * 200
            tile = Image.fromarray(image).resize((222, 167), Image.Resampling.LANCZOS)
            draw.text((x, y + 2), label, fill='white')
            board.paste(tile, (x, y + 20))
    board.save(path)
    return digest(path)


@torch.inference_mode()
def run(prepared, previous, out):
    report = previous / 'report.json'
    if digest(report) != BASELINE:
        raise ValueError('Changed baseline evidence')
    baseline = json.loads(report.read_bytes())
    source = prepared / 'inputs.json'
    if digest(source) != baseline['prepared_sha256'] or algorithm_fingerprint() != baseline['stable_fingerprint']:
        raise ValueError('Changed prepared input or stable implementation')
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != baseline['checkpoint_sha256']:
        raise ValueError('Changed spatial checkpoint')

    inputs = baseline['inputs']
    out.mkdir()
    images, edited, edit_hashes = {}, {}, {}
    for key, spec in inputs.items():
        image = verified(prepared / f'{key}.png', spec['image_sha256'])
        images[key] = image
        edited[key] = stylize(image, np.zeros(image.shape[:2], dtype=bool), EDIT)
        edit_hashes[key] = save(out / f'{key}-edited.png', edited[key])
    cases = {}
    for target_scene, donor in permutations(SCENES):
        for target_light in LIGHTS:
            target_key = f'{target_scene}-{target_light}'
            case_name = f'{target_key}-from-{donor}'
            target = images[target_key]
            expected = finish_spatial(target, edited[target_key], STRENGTH)
            corners = inputs[target_key]['patches']
            masks = make_masks(target, corners)
            expected_hash = save(out / f'{case_name}-expected.png', expected)
            off = baseline['cases'][case_name]['outputs_sha256']
            guide_keys = [f'{donor}-{light}' for light in LIGHTS]
            outputs = {'stable': [], 'spatial13': []}
            measurements = {'stable': {}, 'spatial13': {}}
            artifacts, panels = {}, []
            for guide_key in guide_keys:
                reference = edited[guide_key]
                mapping = analyze(prepared / f'{target_key}.png', out / f'{guide_key}-edited.png')
                frames = {
                    'stable': apply_mapping(target, mapping, STRENGTH),
                    'spatial13': finish_spatial(target, engine.prepare(target, reference).apply(target), STRENGTH),
                }
                for pipeline, on in frames.items():
                    name = f'{target_key}-from-{guide_key}-{pipeline}.png'
                    old = verified(previous / name, off[name])
                    outputs[pipeline].append((old, on))
                    artifacts[name] = save(out / name, on)
                    measurements[pipeline][guide_key] = {
                        label: metrics(target, expected, old, on, mask) for label, mask in masks.items()
                    }
                    panels.append([(f'{guide_key} / input', images[guide_key]), ('Edited reference', reference),
                                   (f'{pipeline} / before edit', old), (f'{pipeline} / after edit', on),
                                   ('Expected same edit on target', expected)])
            for pipeline, ((old6, new6), (old13, new13)) in outputs.items():
                interaction = (new13.astype(np.int16) - old13) - (new6.astype(np.int16) - old6)
                light_scene = {}
                for patch in PATCHES:
                    key = str(patch)
                    input_before = [
                        patch_measurement(images[guide], inputs[guide]['patches'][key]) for guide in guide_keys
                    ]
                    input_after = [
                        patch_measurement(edited[guide], inputs[guide]['patches'][key]) for guide in guide_keys
                    ]
                    before = [patch_measurement(rgb, corners[key]) for rgb in (old6, old13)]
                    after = [patch_measurement(rgb, corners[key]) for rgb in (new6, new13)]
                    light_scene[key] = dict(
                        without_edit=projection(input_before, before),
                        with_edit=projection(input_after, after),
                        output_endpoint_fraction=max(m['endpoint_fraction'] for m in before + after),
                    )
                measurements[pipeline]['lighting'] = light_scene
                measurements[pipeline]['edit_interaction_mae'] = {
                    label: float(np.abs(interaction[mask]).mean()) for label, mask in masks.items()
                }
            cases[case_name] = dict(pipelines=measurements, outputs_sha256=artifacts,
                                   expected_sha256=expected_hash,
                                   panel_sha256=montage(out / f'{case_name}-comparison.png', panels))
    write_json(out / 'report.json', dict(edit=EDIT, baseline_sha256=BASELINE, strength=STRENGTH,
               reference_sha256=edit_hashes, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for argument in ('prepared', 'previous', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
