"""Fit the same Film A renderer with white-only or mixed capture-light samples."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from probe_aligned_cubic_pilot import cubic_mapping, learn
from probe_arri_shared_look import metrics
from probe_chromamill_fixed_material import SURFACES, areas_for
from probe_chromamill_target_light_cross import LEVELS, LIGHTS, PARENT_SHA, SCENES, load_scene
from probe_target_mirror_chroma import BASE, checked_json, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from spatial_render import finish_spatial


ARMS = {'white_only': ('white',), 'mixed_light': tuple(LIGHTS)}
STRENGTH = .8
TRAIN_PIXELS = 8192


def sample_mask(image):
    y, x = np.indices(image.shape[:2])
    return (x // 16 + y // 16) % 2 == 0


def training_pairs(frames, lights, identity=False):
    originals, authored = [], []
    quota = TRAIN_PIXELS // (len(lights) * len(LEVELS))
    for light, level in product(lights, LEVELS):
        original = frames[light, level, 'off']
        edited = original if identity else frames[light, level, 'on']
        available = np.flatnonzero(sample_mask(original))
        if len(available) < quota:
            raise ValueError('Insufficient source support for fixed sample count')
        pixels = available[np.linspace(0, len(available), quota, endpoint=False, dtype=int)]
        originals.append(original.reshape(-1, 3)[pixels])
        authored.append(edited.reshape(-1, 3)[pixels])
    return np.concatenate(originals), np.concatenate(authored)


def fit(frames, lights, identity=False):
    a, b = training_pairs(frames, lights, identity)
    weights = learn(a, b, np.ones(len(a), dtype=bool))
    if weights is None:
        raise ValueError('Could not fit color mapping')
    return cubic_mapping(weights)


def render(original, mapping):
    return finish_spatial(original, apply_mapping(original, mapping, 1), STRENGTH)


def measure(original, expected, output, areas):
    whole = np.ones(original.shape[:2], dtype=bool)
    results = {name: metrics(original[roi], expected[roi], original[roi], output[roi], whole[roi])
               for name, roi in areas.items()}
    return results, all(results[area]['passed'] for area in ('whole', *SURFACES))


def run(out):
    source = BASE / 'chromamill-capture-cross-v1'
    prior = checked_json(source / 'results.json', PARENT_SHA)
    if not all(row['qualified'] for row in prior['checks'].values()):
        raise ValueError('Unqualified physical capture')
    frames = {scene: load_scene(source, scene, prior['prepared'][scene]) for scene in SCENES}
    out.mkdir()
    cases, training, negatives, panels = {}, {}, {}, {}
    passes = {arm: 0 for arm in ARMS}
    blue_improvements, regressions, control_failures = [], [], []
    for scene, reference in (SCENES, SCENES[::-1]):
        donor = frames[reference]
        folder = out / scene
        folder.mkdir()
        tables = {arm: fit(donor, lights) for arm, lights in ARMS.items()}
        neutral = fit(donor, ARMS['mixed_light'], identity=True)
        training[reference] = {arm: {} for arm in ARMS}
        boards = {}
        for light, level in product(LIGHTS, LEVELS):
            tag = f'{light}-{level}'
            checkpoint = donor[light, level, 'off']
            authored = donor[light, level, 'on']
            held = ~sample_mask(checkpoint)
            for arm, table in tables.items():
                fitted = apply_mapping(checkpoint, table, 1)
                errors = np.abs(fitted[held].astype(np.int16) - authored[held])
                training[reference][arm][tag] = {'mae': float(errors.mean()), 'p95': float(np.quantile(errors, .95))}
            original = frames[scene][light, level, 'off']
            expected = finish_spatial(original, frames[scene][light, level, 'on'], STRENGTH)
            negatives[f'{scene}-{tag}'] = bool(np.array_equal(render(original, neutral), original))
            areas = areas_for(original, scene)
            images = {'target': original}
            ratings = {}
            for arm, table in tables.items():
                output = render(original, table)
                images[arm] = output
                results, passed = measure(original, expected, output, areas)
                passes[arm] += int(passed)
                ratings[arm] = dict(regions=results, joint_pass=passed,
                                    boundary_fraction=float(((output == 0) | (output == 255)).any(axis=-1).mean()),
                                    sha256=save(folder / f'{tag}-{arm}.png', output))
            white, mixed = (ratings[arm]['regions'] for arm in ARMS)
            key = f'{scene}-{tag}'
            if light == 'blue' and white['whole']['response_mae'] > 0:
                if mixed['whole']['response_mae'] <= .75 * white['whole']['response_mae']:
                    blue_improvements.append(key)
            regressions.extend(f'{key}:{area}' for area in areas
                               if mixed[area]['response_mae'] > white[area]['response_mae'] + .5)
            if not mixed['colored_control']['passed']:
                control_failures.append(key)
            cases[key] = ratings
            boards[tag] = {**images, 'expected': expected}
        panels[scene] = panel(folder / 'comparison.png', boards)
    passed = (passes['mixed_light'] >= 6 and passes['mixed_light'] > passes['white_only']
              and len(blue_improvements) >= 2 and not regressions and not control_failures and all(negatives.values()))
    json_to(out / 'results.json', dict(parent_sha256=PARENT_SHA, script_sha256=digest(Path(__file__)),
                                      training_pixels=TRAIN_PIXELS, training=training, cases=cases,
                                      panels=panels, joint_pass_counts=passes, blue_improvements=blue_improvements,
                                      regressions=regressions, control_failures=control_failures,
                                      identity=negatives, passed=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
