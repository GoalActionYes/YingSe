"""Measure a known white surface as a two-image color anchor."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked
from background_intervention import target_regions
from fetch_mls_linear_pilot import LIGHTS, SCENES
from independent_coco import ROOT, digest
from probe_mls_light_response import GROUPS, evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_single_reference_material import mean_colors, render
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


SURFACE = 'background_white'


def run(cache, material, out):
    evidence = ROOT/'docs/research/2026-11-18-mls-linear-pilot-evidence.json'
    previous = json.loads(evidence.read_bytes())
    frames, shapes, means = {}, {}, {}
    with checked(material, 'surfaces.zip') as source:
        rows = list(csv.DictReader(io.StringIO(source.read('surfaces/mask_to_surface.csv').decode())))
    with checked(material, 'masks_preview.zip') as masks:
        for scene in SCENES:
            rgb, = [row['mask_colour_rgb'] for row in rows
                    if int(row['scene']) == int(scene) and row['surface'] == SURFACE]
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                palette = np.asarray(image.resize((1157, 950), Image.Resampling.NEAREST))
            shape = (palette == np.array(rgb.split(), dtype=int)).all(-1)
            shapes[scene] = {SURFACE:shape}
            images = {}
            for light in LIGHTS:
                file = cache/f'{scene}_{light}.png'
                assert digest(file) == previous['scenes'][scene]['inputs_sha256'][file.name]
                images[light] = decode_image(file)
                assert images[light].shape == (950, 1157, 3)
            frames[scene] = images
            means[scene] = {light:mean_colors(photo, shapes[scene])[SURFACE] for light,photo in images.items()}

    out.mkdir()
    old, new = LIGHTS
    reported = {}
    for scene in SCENES:
        donor, _ = GROUPS[scene]
        target = frames[scene][old]
        np.testing.assert_array_equal(render(target, shapes[scene], {SURFACE:np.zeros(2)}), target)
        arms = {}
        for arm, light in (('same_setting', old), ('other_swap', new)):
            offset = means[donor][light] - means[scene][old]
            output = render(target, shapes[scene], {SURFACE:offset})
            path = out/f'{scene}-{arm}.png'
            Image.fromarray(output).save(path)
            arms[arm] = dict(sha256=digest(path), offset=offset.tolist())
        reported[scene] = dict(donor=donor, arms=arms)

    proceed = True
    for scene, report in reported.items():
        first, second = (frames[scene][light] for light in LIGHTS)
        truth = finish_spatial(first, second, .8)
        valid = valid_pixels(first, second)
        surface = shapes[scene][SURFACE]
        neutral = target_regions(first, surface)['neutral_proxy'] & surface
        aligned = previous['scenes'][scene]['registration'] >= .5
        regions = dict(white_surface=surface, neutral_subset=neutral)
        report['pixel_support'] = {name:int(mask.sum()) for name,mask in regions.items()}
        for arm, scores in report['arms'].items():
            candidate = decode_image(out/f'{scene}-{arm}.png')
            baseline_file = cache/f'{scene}-{arm}-13.png'
            assert digest(baseline_file) == previous['scenes'][scene]['baseline'][arm]['sha256']
            baseline = decode_image(baseline_file)
            scores['regions'] = {
                name:{method:evaluate(first, output, truth, mask, valid, aligned)
                      for method,output in (('white_anchor', candidate), ('current13', baseline))}
                for name,mask in regions.items()}
        checks = {}
        for name in regions:
            neg = report['arms']['same_setting']['regions'][name]['white_anchor']
            pair = report['arms']['other_swap']['regions'][name]
            pos, base = pair['white_anchor'], pair['current13']
            gain = 1-pos['error']/base['error'] if pos['eligible'] and base['error'] else None
            passed = None if gain is None else bool(neg['same_setting_pass'] and pos['lamp_pass'] and gain>=.15)
            checks[name] = dict(gain=gain, passed=passed)
        report['gates'] = checks
        proceed &= checks['white_surface']['passed'] is True and all(
            item['passed'] is not False for item in checks.values())
    data = dict(scenes=reported, proceeded=proceed, baseline_evidence_sha256=digest(evidence),
                source_sha256=digest(Path(__file__)), identity=True)
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({scene:row['gates'] for scene,row in reported.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'material', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.cache, args.material, args.out)
