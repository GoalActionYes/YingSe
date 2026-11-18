"""Measure fixed-gain coverage and run the unchanged 13 baseline if eligible."""
import argparse
import csv
import io
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image
import torch

from audit_fau_two_light_intake import native_rgb
from audit_mls_material_identity import checked
from background_intervention import target_regions
from fetch_mls_linear_pilot import LIGHTS, SCENES
from independent_coco import digest
from probe_fau_two_light_response import encode
from probe_mls_light_response import GROUPS, evaluate, registration
from spatial_render import SpatialEngine, finish_spatial


def read(path, witness):
    raw = path.read_bytes()
    assert digest(path) == witness['sha256'] and zlib.crc32(raw) == witness['crc32']
    assert raw[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR'
    width, height, depth, channels = struct.unpack('>IIBB', raw[16:26])
    assert depth == 16 and channels == 2 and 1 <= width <= 1300 and 1 <= height <= 1100
    return native_rgb(raw, height, width)


def valid_pixels(first, second):
    return ((first >= 16) & (first <= 239) & (second >= 16) & (second <= 239)).all(axis=-1)


def run(inputs, material, out):
    torch.set_num_threads(4)
    intake = json.loads((inputs/'intake.json').read_bytes())
    old_light, new_light = LIGHTS
    out.mkdir()
    with checked(material, 'surfaces.zip') as spectra:
        rows = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode('utf-8-sig'))))
    report, prepared = {}, {}
    with checked(material, 'masks_preview.zip') as masks:
        for scene in SCENES:
            filenames = [f'{scene}_{light}.png' for light in LIGHTS]
            raw = [read(inputs/name, intake['entries'][name]) for name in filenames]
            assert raw[0].shape == raw[1].shape
            height, width = raw[0].shape[:2]
            scaled = [frame.astype(np.uint32)*4 for frame in raw]
            no_gain = [encode(frame) for frame in raw]
            frames = [encode(frame.clip(0, 65535)) for frame in scaled]
            first, second = frames
            for name, frame in zip(filenames, frames):
                Image.fromarray(frame).save(out/name)
            alignment = registration(first, second)
            desired = finish_spatial(first, second, .8)
            valid = valid_pixels(first, second)
            orig_valid = valid_pixels(*no_gain)
            common_regions = target_regions(first, np.zeros(first.shape[:2], bool))
            regions = {name: common_regions[name] for name in ('whole', 'neutral_proxy')}
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as mask:
                assert abs((width/height)/(mask.width/mask.height)-1) < .002
                palette = np.asarray(mask.resize((width,height), Image.Resampling.NEAREST))
            for name in GROUPS[scene][1]:
                rgb, = [row['mask_colour_rgb'] for row in rows
                        if int(row['scene']) == int(scene) and row['surface'] == name]
                regions[name] = (palette == np.array(rgb.split(), dtype=int)).all(axis=-1)
            coverage = {name: evaluate(first, first, desired, region, valid, alignment >= .5)
                        for name, region in regions.items()}
            for name, region in regions.items():
                coverage[name]['unity_valid_pixels'] = int((region & orig_valid).sum())
            prepared[scene] = (first, second, desired, valid, regions, alignment)
            report[scene] = dict(registration=alignment, dimensions=[width, height], coverage=coverage,
                                 source_max_channel=[frame.max(axis=(0, 1)).tolist() for frame in raw],
                                 source_saturated=[int((frame == 65535).any(axis=-1).sum()) for frame in raw],
                                 gain_clips=[int((frame > 65535).any(axis=-1).sum()) for frame in scaled],
                                 inputs_sha256={name: digest(out/name) for name in filenames})
    qualified = sum(region['eligible'] for data in report.values() for region in data['coverage'].values())
    checkpoint = None
    if qualified >= 2:
        engine = SpatialEngine('multiref-v1')
        checkpoint = engine.checkpoint_sha
        for scene, data in report.items():
            first, second, desired, valid, regions, alignment = prepared[scene]
            other = GROUPS[scene][0]
            old, new, *_ = prepared[other]
            data['baseline'] = {}
            for arm, reference in (('same_setting', old), ('other_swap', new), ('same_scene_swap', second)):
                output = finish_spatial(first, engine.prepare(first, reference).apply(first), .8)
                file = out/f'{scene}-{arm}-13.png'
                Image.fromarray(output).save(file)
                data['baseline'][arm] = dict(sha256=digest(file), regions={
                    name: evaluate(first, output, desired, region, valid, alignment >= .5)
                    for name, region in regions.items()})
    result = dict(scenes=report, qualified_regions=qualified, checkpoint_sha256=checkpoint,
                  intake_sha256=digest(inputs/'intake.json'), source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(qualified=qualified, baseline=checkpoint is not None)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'material', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.inputs, args.material, args.out)
