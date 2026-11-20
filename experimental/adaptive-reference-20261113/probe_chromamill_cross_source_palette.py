"""Add independent captured colors or known full-gamut colors to a frozen fit."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from probe_arri_sdr_renderability import transforms
from probe_capture_invariant_look import render_style
from probe_chromamill_fixed_material import areas_for
from probe_chromamill_local_fit_residual import MIXED_SHA, correct, learn_residual, support
from probe_chromamill_mixed_light_fit import ARMS, STRENGTH, fit, measure, render, training_pairs
from probe_chromamill_target_light_cross import LEVELS, LIGHTS, PARENT_SHA, SCENES, load_scene
from probe_mls_film_light_cross import SOURCE_SHA
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial


MLS_EDIT_SHA = 'add480d14eeee078b9108c4a1575719199637010bdacc92f21de0717ffe34754'
RESIDUAL_SHA = '9f05ef4b1949120fed9051fac9d7a793e6a483cec799d2565a013a4f9704c444'


def banks(look_hash):
    inputs = BASE / 'mls-linear-pilot'
    edits = BASE / 'mls-film-light-cross-v1'
    captures = checked_json(inputs / 'results.json', SOURCE_SHA)
    edited = checked_json(edits / 'results.json', MLS_EDIT_SHA)
    cube = BASE / 'arri-shared-look-assets-v1/film-a.cube'
    if edited['source_sha256'] != SOURCE_SHA or edited['lut_sha256'] != look_hash or digest(cube) != look_hash:
        raise ValueError('Different source or Film A operation')
    left, right, counts = [], [], {}
    for scene, (light, suffix) in product(('02', '03'), (('standard', ''), ('blue', '_LED-B025'))):
        raw_name = f'{scene}_2HAL_DESK{suffix}.png'
        original = checked_rgb(inputs / raw_name, captures['scenes'][scene]['inputs_sha256'][raw_name])
        style = checked_rgb(edits / f'{scene}-{light}-reference-film.png', edited['references'][scene][light])
        if original.shape != style.shape:
            raise ValueError('MLS pair geometry changed')
        palette, first, inverse = np.unique(original.reshape(-1, 3), return_index=True, return_inverse=True, axis=0)
        desired = style.reshape(-1, 3)[first]
        np.testing.assert_array_equal(desired[inverse].reshape(original.shape), style)
        left.append(palette)
        right.append(desired)
        counts[f'{scene}-{light}'] = len(palette)
    axis = np.rint(np.linspace(0, 255, 64)).astype(np.uint8)
    uniform = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    return {'mls': (np.concatenate(left), np.concatenate(right)),
            'uniform_control': (uniform, render_style(uniform, transforms(cube)))}, counts


def run(archive, out):
    capture = BASE / 'chromamill-capture-cross-v1'
    source = checked_json(capture / 'results.json', PARENT_SHA)
    previous = checked_json(archive / 'results.json', RESIDUAL_SHA)
    mixed_archive = checked_json(BASE / 'chromamill-mixed-light-fit-v1/results.json', MIXED_SHA)
    inputs = {scene: load_scene(capture, scene, source['prepared'][scene]) for scene in SCENES}
    extra, counts = banks(source['look_sha256'])
    out.mkdir()
    cases, panels, palettes = {}, {}, {}
    checks = {name: [] for name in extra}
    for scene, reference in (SCENES, SCENES[::-1]):
        folder = out / scene
        folder.mkdir()
        donor = inputs[reference]
        table = fit(donor, ARMS['mixed_light'])
        original_pairs = training_pairs(donor, ARMS['mixed_light'])
        old_palette = learn_residual(*original_pairs, table)
        white, _ = training_pairs(donor, ARMS['white_only'])
        white_tree = cKDTree(np.unique(white, axis=0))
        identity_table = fit(donor, ARMS['mixed_light'], identity=True)
        augmented, identities = {}, {}
        for name, (left, right) in extra.items():
            a, b = [np.concatenate((prior, added)) for prior, added in zip(original_pairs, (left, right), strict=True)]
            augmented[name] = learn_residual(a, b, table)
            identities[name] = learn_residual(a, a, identity_table)
        palettes[scene] = {name: item[0].n for name, item in augmented.items()}
        board = {}
        for light, level in product(LIGHTS, LEVELS):
            tag = f'{light}-{level}'
            key = f'{scene}-{tag}'
            target = inputs[scene][light, level, 'off']
            old = previous['cases'][key]
            frozen = checked_rgb(archive / scene / f'{tag}-local.png', old['sha256'])
            uncorrected = render(target, table)
            np.testing.assert_array_equal(correct(target, table, old_palette, white_tree)[0], frozen)
            expected = finish_spatial(target, inputs[scene][light, level, 'on'], STRENGTH)
            areas = areas_for(target, scene)
            variants = {}
            images = {'target': target, 'old residual': frozen}
            for name, palette in augmented.items():
                candidate, details = correct(target, table, palette, white_tree)
                unchanged = details['weights'] == 0
                np.testing.assert_array_equal(candidate[unchanged], uncorrected[unchanged])
                identity = np.array_equal(correct(target, identity_table, identities[name], white_tree)[0], target)
                ratings, joint = measure(target, expected, candidate, areas)
                for region, scores in ratings.items():
                    if not scores['passed'] or scores['response_mae'] > old['regions'][region]['response_mae'] + .5:
                        checks[name].append(f'{key}:{region}')
                if light == 'blue':
                    white_mae = mixed_archive['cases'][key]['white_only']['regions']['whole']['response_mae']
                    if ratings['whole']['response_mae'] > .75 * white_mae:
                        checks[name].append(f'{key}:lost_blue_gain')
                if not identity:
                    checks[name].append(f'{key}:identity')
                coverage = {region: support(details, area)['mixed']['eighth_within_six']
                            for region, area in areas.items()}
                weight = {region: float(details['weights'][area].mean()) for region, area in areas.items()}
                variants[name] = dict(scores=ratings, joint=joint, identity=bool(identity),
                                      coverage=coverage, weight=weight,
                                      sha256=save(folder / f'{tag}-{name}.png', candidate))
                images[name] = candidate
            cases[key] = variants
            board[tag] = {**images, 'expected': expected}
        panels[scene] = panel(folder / 'comparison.png', board)
    json_to(out / 'results.json', dict(source_sha256=PARENT_SHA, old_sha256=RESIDUAL_SHA,
                                      mls_sha256=MLS_EDIT_SHA, script_sha256=digest(Path(__file__)),
                                      mls_sample_counts=counts, uniform_colors=len(extra['uniform_control'][0]),
                                      palette_counts=palettes, failures=checks, panels=panels, cases=cases,
                                      passed={name: not failures for name, failures in checks.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
