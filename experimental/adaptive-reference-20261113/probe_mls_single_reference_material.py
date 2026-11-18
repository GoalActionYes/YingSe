"""Test two-image color differences with offline known material identities."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked
from fetch_mls_linear_pilot import LIGHTS, SCENES
from independent_coco import ROOT, digest
from probe_fau_illuminant_replay import replay
from probe_mls_light_response import GROUPS, evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_material_response import light_gain
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def mean_colors(rgb, regions):
    valid = valid_pixels(rgb, rgb)
    means = {}
    for name, region in regions.items():
        selected = rgb[valid & region]
        assert len(selected) >= 256
        logs = np.log(linear_rgb(selected))
        means[name] = (logs[:, [0, 2]]-logs[:, 1:2]).mean(axis=0)
    return means


def render(target, regions, offsets):
    result = target.copy()
    for name, region in regions.items():
        changed, _ = replay(target, light_gain(offsets[name]), region)
        result[region] = changed[region]
    return result


def run(cache, material, out):
    evidence = ROOT/'docs/research/2026-11-18-mls-linear-pilot-evidence.json'
    earlier = json.loads(evidence.read_bytes())
    assert earlier['qualified_regions'] == 8
    frames, regions, stats = {}, {}, {}
    with checked(material, 'surfaces.zip') as spectra:
        tables = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode())))
    with checked(material, 'masks_preview.zip') as masks:
        for scene in SCENES:
            images = {}
            for light in LIGHTS:
                path = cache/f'{scene}_{light}.png'
                assert digest(path) == earlier['scenes'][scene]['inputs_sha256'][path.name]
                images[light] = decode_image(path)
                assert images[light].shape == (950, 1157, 3)
            frames[scene] = images
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as source:
                mask = np.asarray(source.resize((1157,950), Image.Resampling.NEAREST))
            known = {}
            for name in GROUPS[scene][1]:
                color, = [row['mask_colour_rgb'] for row in tables
                          if row['surface'] == name and int(row['scene']) == int(scene)]
                known[name] = (mask == np.array(color.split(), dtype=int)).all(axis=-1)
            regions[scene] = known
            stats[scene] = {light:mean_colors(image, known) for light,image in images.items()}

    old, new = LIGHTS
    report = {}
    out.mkdir()
    for scene, (donor, names) in GROUPS.items():
        if scene not in SCENES:
            continue
        target = frames[scene][old]
        same = stats[scene][old]
        identity = render(target, regions[scene], {name:same[name]-same[name] for name in names})
        np.testing.assert_array_equal(identity, target)
        cases = {}
        for arm, reference_light in (('same_setting', old), ('other_swap', new)):
            reference = stats[donor][reference_light]
            offsets = {name:reference[name]-same[name] for name in names}
            mean = np.mean(list(offsets.values()), axis=0)
            versions = dict(matched=offsets, pooled={name:mean for name in names})
            results = {}
            for method, values in versions.items():
                rendered = render(target, regions[scene], values)
                file = out/f'{scene}-{arm}-{method}.png'
                Image.fromarray(rendered).save(file)
                results[method] = dict(sha256=digest(file), offsets={name:value.tolist() for name,value in values.items()})
            cases[arm] = results
        report[scene] = dict(donor=donor, cases=cases)

    for scene, entry in report.items():
        target, switched = (frames[scene][light] for light in LIGHTS)
        truth = finish_spatial(target, switched, .8)
        valid = valid_pixels(target, switched)
        for arm, cases in entry['cases'].items():
            for method, record in cases.items():
                image = decode_image(out/f'{scene}-{arm}-{method}.png')
                record['regions'] = {name:evaluate(target, image, truth, mask, valid, True)
                                     for name,mask in regions[scene].items()}
        negative = entry['cases']['same_setting']['matched']['regions']
        positive = entry['cases']['other_swap']['matched']['regions']
        control = entry['cases']['other_swap']['pooled']['regions']
        entry['gates'] = {}
        for name, result in positive.items():
            gain = 1-result['error']/control[name]['error']
            entry['gates'][name] = dict(gain=gain, passed=bool(negative[name]['same_setting_pass']
                                                             and result['lamp_pass'] and gain>=.15))

    proceed = all(check['passed'] for row in report.values() for check in row['gates'].values())
    result = dict(scenes=report, proceeded=proceed, identity=True, baseline_sha256=digest(evidence),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({scene:entry['gates'] for scene,entry in report.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'material', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.cache, args.material, args.out)
