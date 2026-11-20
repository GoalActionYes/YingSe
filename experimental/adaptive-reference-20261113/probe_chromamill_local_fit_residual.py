"""Test a coverage-gated local correction to the frozen color fit."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from probe_arri_paired_coverage import NEIGHBORS, RADIUS, mapping
from probe_chromamill_mixed_light_fit import ARMS, STRENGTH, fit, measure, render, training_pairs
from probe_chromamill_fixed_material import areas_for
from probe_chromamill_target_light_cross import LEVELS, LIGHTS, PARENT_SHA, SCENES, load_scene
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial


MIXED_SHA = '5d3c122992c267b15db6a8e0d3ec850af2a9eed56fe568b8552944d0a3b8581e'


def learn_residual(before, after, table):
    palette, look_delta = mapping(before, after)
    fitted = apply_mapping(palette[:, None, :], table, 1)[:, 0]
    residual = look_delta - (fitted.astype(float) - palette)
    return cKDTree(palette), residual


def correct(target, base_lut, mixed, white):
    palette, indices = np.unique(target.reshape(-1, 3), axis=0, return_inverse=True)
    tree, residual = mixed
    distances, neighbors = tree.query(palette, k=NEIGHBORS, workers=1)
    white_distances, _ = white.query(palette, k=NEIGHBORS, workers=1)
    weights = 1 / np.maximum(distances, 1)
    average = (weights[..., None] * residual[neighbors]).sum(axis=1) / weights.sum(axis=1)[:, None]
    taper = np.clip((RADIUS - distances[:, -1]) / (RADIUS / 2), 0, 1)
    base = apply_mapping(palette[:, None, :], base_lut, 1)[:, 0]
    adjusted = np.rint(np.clip(base + taper[:, None] * average, 0, 255)).astype(np.uint8)
    image = adjusted[indices].reshape(target.shape)
    metadata = {
        'white': white_distances[indices][:, (0, -1)].reshape(*target.shape[:2], 2),
        'mixed': distances[indices][:, (0, -1)].reshape(*target.shape[:2], 2),
        'weights': taper[indices].reshape(target.shape[:2]),
    }
    return finish_spatial(target, image, STRENGTH), metadata


def support(metadata, area):
    return {
        arm: {'median_first_eighth': np.median(metadata[arm][area].reshape(-1, 2), axis=0).tolist(),
              'first_within_six': float((metadata[arm][area][..., 0] < RADIUS).mean()),
              'eighth_within_six': float((metadata[arm][area][..., 1] < RADIUS).mean())}
        for arm in ('white', 'mixed')
    } | {'mean_correction_weight': float(metadata['weights'][area].mean())}


def run(archive, out):
    path = BASE / 'chromamill-capture-cross-v1'
    source = checked_json(path / 'results.json', PARENT_SHA)
    prior = checked_json(archive / 'results.json', MIXED_SHA)
    photos = {scene: load_scene(path, scene, source['prepared'][scene]) for scene in SCENES}
    out.mkdir()
    cases, panels = {}, {}
    failures = []
    for scene, reference in (SCENES, SCENES[::-1]):
        folder = out / scene
        folder.mkdir()
        donor = photos[reference]
        tables = {arm: fit(donor, lights) for arm, lights in ARMS.items()}
        before, after = training_pairs(donor, ARMS['mixed_light'])
        mixed = learn_residual(before, after, tables['mixed_light'])
        plain, _ = training_pairs(donor, ARMS['white_only'])
        white = cKDTree(np.unique(plain, axis=0))
        neutral_lut = fit(donor, ARMS['mixed_light'], identity=True)
        neutral = learn_residual(before, before, neutral_lut)
        images = {}
        for light, level in product(LIGHTS, LEVELS):
            tag = f'{light}-{level}'
            key = f'{scene}-{tag}'
            original = photos[scene][light, level, 'off']
            archived = prior['cases'][key]
            renders = {}
            for arm, table in tables.items():
                saved = checked_rgb(archive / scene / f'{tag}-{arm}.png', archived[arm]['sha256'])
                np.testing.assert_array_equal(render(original, table), saved)
                renders[arm] = saved
            corrected, metadata = correct(original, tables['mixed_light'], mixed, white)
            identity, _ = correct(original, neutral_lut, neutral, white)
            supported = metadata['weights'] > 0
            np.testing.assert_array_equal(corrected[~supported], renders['mixed_light'][~supported])
            expected = finish_spatial(original, photos[scene][light, level, 'on'], STRENGTH)
            areas = areas_for(original, scene)
            scores, joint = measure(original, expected, corrected, areas)
            identity_pass = np.array_equal(original, identity)
            for region, values in scores.items():
                if not values['passed'] or values['response_mae'] > archived['mixed_light']['regions'][region]['response_mae'] + .5:
                    failures.append(f'{key}:{region}')
            if light == 'blue' and scores['whole']['response_mae'] > .75 * archived['white_only']['regions']['whole']['response_mae']:
                failures.append(f'{key}:lost_blue_gain')
            if not identity_pass:
                failures.append(f'{key}:identity')
            cases[key] = dict(regions=scores, joint_pass=joint, identity_pass=identity_pass,
                              support={region: support(metadata, area) for region, area in areas.items()},
                              sha256=save(folder / f'{tag}-local.png', corrected))
            images[tag] = {'original': original, 'frozen mixed': renders['mixed_light'],
                           'local correction': corrected, 'expected': expected}
        panels[scene] = panel(folder / 'comparison.png', images)
    json_to(out / 'results.json', dict(archive_sha256=MIXED_SHA, source_sha256=PARENT_SHA,
                                       script_sha256=digest(Path(__file__)), cases=cases, panels=panels,
                                       failures=failures, passed=not failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
