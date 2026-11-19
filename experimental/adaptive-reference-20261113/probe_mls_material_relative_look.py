"""Test known material correspondences as evidence of an authored global look."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked, table
from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_capture_invariant_look import render_style, style_transforms
from probe_capture_look_inverse_prior import inverse_tables
from probe_mls_film_light_cross import SOURCE_SHA
from probe_mls_light_response import GROUPS
from probe_multilum_chromaticity import linear_rgb
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, json_to, save
from spatial_render import finish_spatial


SCENES = ('02', '03')
LIGHTS = {'standard': '', 'blue': '_LED-B025'}
HISTORY_SHA = 'add480d14eeee078b9108c4a1575719199637010bdacc92f21de0717ffe34754'


def load_material_regions():
    folder = BASE / 'mls-material-identity'
    with checked(folder, 'surfaces.zip') as surfaces:
        rows = table(surfaces, 'surfaces/mask_to_surface.csv')
    regions = {}
    with checked(folder, 'masks_preview.zip') as archive:
        for scene in SCENES:
            with Image.open(BytesIO(archive.read(f'masks_preview/{scene}.png'))) as image:
                palette = np.asarray(image.resize((1157, 950), Image.Resampling.NEAREST))
            matched = []
            for name in GROUPS[scene][1]:
                code, = [row['mask_colour_rgb'] for row in rows
                         if int(row['scene']) == int(scene) and row['surface'] == name]
                matched.append((palette == np.array(code.split(), dtype=int)).all(axis=-1))
            regions[scene] = matched
    return regions


def material_pixels(image, regions):
    valid = ((image >= 16) & (image <= 239)).all(axis=-1)
    return [image[valid & area] for area in regions]


def relative_chroma(colors):
    values = [np.log(np.maximum(linear_rgb(values), 1 / (255 * 12.92))) for values in colors]
    first, second = [(logs[:, [0, 2]] - logs[:, 1:2]).mean(axis=0) for logs in values]
    return first - second


def choose(target_colors, reference, regions, lattice, inverse):
    selected = material_pixels(reference, regions)
    counts = [len(group) for group in selected]
    if min(counts) < 256:
        return {'on': False, 'pixels': counts, 'costs': None}
    inputs = np.concatenate(selected)
    tree, graded = inverse
    index = tree.query(inputs, workers=1)[1]
    split = counts[0]
    restored = (lattice[index[:split]], lattice[index[split:]])
    reference_vectors = [relative_chroma(selected), relative_chroma(restored)]
    costs = [float(np.sqrt(np.square(vector - target_colors).mean())) for vector in reference_vectors]
    cycle = float(np.abs(graded[index].astype(np.int16) - inputs).mean())
    return {'on': costs[0] - costs[1] >= .025 and cycle <= 2, 'pixels': counts,
            'costs': costs, 'inverse_mae': cycle}


def run(out):
    source = BASE / 'mls-linear-pilot'
    frozen = checked_json(source / 'results.json', SOURCE_SHA)
    history_dir = BASE / 'mls-film-light-cross-v1'
    history = checked_json(history_dir / 'results.json', HISTORY_SHA)
    regions = load_material_regions()
    processors = style_transforms(BASE)['film_a']
    lattice, tables = inverse_tables({'film_a': processors})
    frames = {}
    for scene in SCENES:
        frames[scene] = {}
        for light, suffix in LIGHTS.items():
            name = f'{scene}_2HAL_DESK{suffix}.png'
            original = checked_rgb(source / name, frozen['scenes'][scene]['inputs_sha256'][name])
            edited = checked_rgb(history_dir / f'{scene}-{light}-reference-film.png',
                                 history['references'][scene][light])
            np.testing.assert_array_equal(render_style(original, processors), edited)
            frames[scene][light] = {'off': original, 'on': edited}

    out.mkdir()
    cases = {}
    for scene in SCENES:
        donor = GROUPS[scene][0]
        target, edited = frames[scene]['standard'].values()
        expected = finish_spatial(target, edited, STRENGTH)
        known_output = finish_spatial(target, render_style(target, processors), STRENGTH)
        np.testing.assert_array_equal(known_output, expected)
        target_pixels = material_pixels(target, regions[scene])
        if min(map(len, target_pixels)) < 256:
            raise ValueError('Target material is below its minimum support')
        target_colors = relative_chroma(target_pixels)
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(x, ~whole)['neutral_proxy'] for x in (target, edited)))
        if int(neutral.sum()) != history['cases'][scene]['neutral_pixels']:
            raise ValueError('Changed neutral mask support')
        views, outputs = {}, {}
        for light, states in frames[donor].items():
            outputs[light], decisions = {}, {}
            for state, reference in states.items():
                result = choose(target_colors, reference, regions[donor], lattice, tables['film_a'])
                frame = known_output if result['on'] else target
                result['sha256'] = save(out / f'{scene}-{light}-{state}.png', frame)
                outputs[light][state] = frame
                decisions[state] = result
            off, on = outputs[light].values()
            views[light] = {'decisions': decisions, 'response': {
                label: metrics(target, expected, off, on, mask)
                for label, mask in (('whole', whole), ('neutral', neutral))}}
        spill = {state: float(np.abs(outputs['standard'][state][neutral].astype(float)
                                     - outputs['blue'][state][neutral]).mean()) for state in ('off', 'on')}
        choices_ok = all(not row['decisions']['off']['on'] and row['decisions']['on']['on']
                         for row in views.values())
        passed = choices_ok and max(spill.values()) <= 3 and all(
            score['passed'] for row in views.values() for score in row['response'].values())
        cases[scene] = {'views': views, 'neutral_spill': spill, 'passed': passed}
    json_to(out / 'results.json', {'cases': cases, 'parent_sha256': HISTORY_SHA,
                                   'passed': all(case['passed'] for case in cases.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
