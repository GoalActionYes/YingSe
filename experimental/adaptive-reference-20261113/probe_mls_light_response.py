"""Measure 13's response to MLS reference scene and lamp changes."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from audit_mls_material_identity import checked
from background_intervention import target_regions
from fetch_mls_light_pilot import LIGHTS, SCENES
from independent_coco import digest
from probe_multilum_chromaticity import differences
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


GROUPS = {
    '02': ('03', ('paper_prism_blue', 'paper_cube_green')),
    '03': ('02', ('paper_prism_blue', 'paper_cube_green')),
    '13': ('22', ('wood_board_dark', 'wood_board_light')),
    '22': ('13', ('wood_board_dark', 'wood_board_light')),
}


def registration(original, lit):
    gradients = []
    for image in (original, lit):
        y = image.astype(float) @ np.array([.299, .587, .114])
        gradients.append(np.concatenate((np.diff(y, axis=0).ravel(), np.diff(y, axis=1).ravel())))
    first, second = gradients
    return float((first @ second)/(np.linalg.norm(first)*np.linalg.norm(second)))


def evaluate(target, output, truth, area, valid, aligned):
    size = int(area.sum())
    selected = area & valid
    count = int(selected.sum())
    if not count:
        return dict(eligible=False, pixels=0, coverage=0.)
    change, _ = differences(target[selected], output[selected])
    desired, _ = differences(target[selected], truth[selected])
    observed = float(np.mean(desired**2)**.5)
    error = float(np.mean((change-desired)**2)**.5)
    effect = float(np.mean(change**2)**.5)
    power = float(np.sum(desired**2))
    projection = float(np.sum(change*desired)/power) if power else None
    coverage = count/size
    eligible = aligned and count >= 256 and coverage >= .20 and observed >= .05
    return dict(eligible=eligible, pixels=count, coverage=coverage, target_rms=observed,
                error=error, induced_rms=effect, projection=projection,
                lamp_pass=bool(eligible and projection >= .50 and error <= .90*observed),
                same_setting_pass=bool(count >= 256 and coverage >= .20 and effect <= .05))


def run(inputs, material, out):
    torch.set_num_threads(4)
    intake = json.loads((inputs/'intake.json').read_bytes())
    photos = {}
    for scene in SCENES:
        for light in LIGHTS:
            name = f'{scene}_{light}.jpg'
            path = inputs/name
            assert digest(path) == intake[name]['sha256']
            photo = decode_image(path)
            assert photo.shape == (493, 600, 3)
            photos[scene, light] = photo

    old_light, new_light = LIGHTS
    out.mkdir()
    engine = SpatialEngine('multiref-v1')
    report, renders = {}, {}
    for target_id, (other, _) in GROUPS.items():
        target = photos[target_id, old_light]
        arms = dict(same_setting=(other, old_light), other_swap=(other, new_light),
                    same_scene_swap=(target_id, new_light))
        records = {}
        for arm, key in arms.items():
            artifact = finish_spatial(target, engine.prepare(target, photos[key]).apply(target), .8)
            path = out/f'{target_id}-{arm}.png'
            Image.fromarray(artifact).save(path)
            records[arm] = dict(sha256=digest(path), reference=list(key))
            renders[target_id, arm] = artifact
        report[target_id] = dict(arms=records)

    with checked(material, 'surfaces.zip') as spectra:
        csv_data = spectra.read('surfaces/mask_to_surface.csv').decode('utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(csv_data)))
    with checked(material, 'masks_preview.zip') as archive:
        for scene, row in report.items():
            target, lit = photos[scene, old_light], photos[scene, new_light]
            row['registration'] = registration(target, lit)
            valid = (((target >= 16) & (target <= 239) & (lit >= 16) & (lit <= 239)).all(axis=-1))
            truth = finish_spatial(target, lit, .8)
            with Image.open(io.BytesIO(archive.read(f'masks_preview/{scene}.png'))) as image:
                assert image.size == (578, 475)
                mask_rgb = np.array(image.resize((600, 493), Image.Resampling.NEAREST))
            regions = target_regions(target, np.zeros(target.shape[:2], bool))
            areas = {key: regions[key] for key in ('whole', 'neutral_proxy')}
            for name in GROUPS[scene][1]:
                color, = [entry['mask_colour_rgb'] for entry in rows
                          if int(entry['scene']) == int(scene) and entry['surface'] == name]
                areas[name] = (mask_rgb == np.array(color.split(), dtype=int)).all(axis=-1)
                assert areas[name].sum() >= 256
            for arm, record in row['arms'].items():
                record['regions'] = {key: evaluate(target, renders[scene, arm], truth, area, valid,
                                                   row['registration'] >= .5)
                                     for key, area in areas.items()}
    data = dict(scenes=report, source_sha256=digest(Path(__file__)), input_manifest_sha256=digest(inputs/'intake.json'),
                checkpoint_sha256=engine.checkpoint_sha, input_members=intake)
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'material', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.inputs, args.material, args.out)
