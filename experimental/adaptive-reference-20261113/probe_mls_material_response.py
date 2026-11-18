"""Test cross-scene light replay conditioned on measured MLS surface identity."""
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
from probe_multilum_chromaticity import differences
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def light_gain(chroma):
    return np.exp([chroma[0], 0, chroma[1]])


def run(cache, material, out):
    evidence = ROOT/'docs/research/2026-11-18-mls-linear-pilot-evidence.json'
    previous = json.loads(evidence.read_bytes())
    assert previous['qualified_regions'] == 8
    scenes, regions = {}, {}
    with checked(material, 'surfaces.zip') as spectra:
        rows = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode('utf-8-sig'))))
    with checked(material, 'masks_preview.zip') as masks:
        for scene in SCENES:
            files = [cache/f'{scene}_{light}.png' for light in LIGHTS]
            assert all(digest(path) == previous['scenes'][scene]['inputs_sha256'][path.name] for path in files)
            before, after = map(decode_image, files)
            assert before.shape == after.shape == (950, 1157, 3)
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                palette = np.asarray(image.resize((1157, 950), Image.Resampling.NEAREST))
            scenes[scene] = before, after
            regions[scene] = {}
            for name in GROUPS[scene][1]:
                color, = [row['mask_colour_rgb'] for row in rows if row['surface'] == name and int(row['scene']) == int(scene)]
                regions[scene][name] = (palette == np.array(color.split(), dtype=int)).all(axis=-1)
    out.mkdir()
    fits = {}
    for scene, (before, after) in scenes.items():
        valid = valid_pixels(before, after)
        sample = {}
        for name, region in regions[scene].items():
            selected = valid & region
            values, _ = differences(before[selected], after[selected])
            identity, _ = differences(before[selected], before[selected])
            np.testing.assert_array_equal(identity, np.zeros_like(identity))
            assert len(values) == previous['scenes'][scene]['coverage'][name]['pixels']
            sample[name] = (values.mean(axis=0), len(values))
        fits[scene] = sample

    results = {}
    for scene, (before, after) in scenes.items():
        donor, names = GROUPS[scene]
        first, second = names
        donor_fits = fits[donor]
        average = np.average([donor_fits[name][0] for name in names], axis=0,
                             weights=[donor_fits[name][1] for name in names])
        cross = {name: donor_fits[name][0] for name in names}
        color_maps = dict(matched=cross, crossed={first:cross[second], second:cross[first]},
                          pooled={name:average for name in names}, zero={name:np.zeros(2) for name in names})
        result = {}
        for arm, offsets in color_maps.items():
            rendered = before.copy()
            for name, coeff in offsets.items():
                mask = regions[scene][name]
                corrected, _ = replay(before, light_gain(coeff), mask)
                rendered[mask] = corrected[mask]
            if arm == 'zero':
                np.testing.assert_array_equal(rendered, before)
            else:
                file = out/f'{scene}-{arm}.png'
                Image.fromarray(rendered).save(file)
                truth = finish_spatial(before, after, .8)
                valid = valid_pixels(before, after)
                result[arm] = dict(sha256=digest(file), regions={
                    name:evaluate(before, rendered, truth, region, valid, True)
                    for name, region in regions[scene].items()})
        wins = {}
        for name in names:
            matched = result['matched']['regions'][name]
            gains = {arm:1-matched['error']/result[arm]['regions'][name]['error'] for arm in ('pooled', 'crossed')}
            wins[name] = dict(improvement=gains, passed=matched['lamp_pass'] and min(gains.values()) >= .15)
        results[scene] = dict(donor=donor, arms=result, gates=wins)
    proceeded = all(check['passed'] for row in results.values() for check in row['gates'].values())
    report = dict(scenes=results, fitted_chroma={scene:{name:vector.tolist() for name,(vector,_) in values.items()}
                                              for scene, values in fits.items()},
                  proceeded=proceeded, zero_identity=True, previous_sha256=digest(evidence),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({scene:record['gates'] for scene,record in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'material', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.cache, args.material, args.out)
