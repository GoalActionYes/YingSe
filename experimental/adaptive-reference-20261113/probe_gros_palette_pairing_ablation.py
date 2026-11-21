"""Isolate whether a joint color distribution can replace registered edit pairs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, finish_spatial, json_to, save


SAMPLES = 1024
METHODS = ('registered', 'distribution')
STATES = ('off', 'on')


def sample_indices(frame):
    colors = frame.reshape(-1, 3)
    key = colors.astype(np.uint32) @ np.array([65536, 256, 1], dtype=np.uint32)
    ordered = np.argsort(key, kind='stable')
    middle = (np.arange(SAMPLES) * 2 + 1) * len(ordered) // (2 * SAMPLES)
    return ordered[middle]


def learn(original, image, method):
    position = sample_indices(original)
    source = original.reshape(-1, 3)[position].astype(np.float64)
    if method == 'registered':
        destination = image.reshape(-1, 3)[position].astype(np.float64)
    else:
        independent = image.reshape(-1, 3)[sample_indices(image)].astype(np.float64)
        cost = np.square(source[:, None, :] - independent[None, :, :]).sum(axis=-1)
        rows, columns = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(SAMPLES)):
            raise ValueError('Incomplete palette assignment')
        destination = independent[columns]
    unique, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
    if len(unique) < 8:
        raise ValueError('Insufficient palette support')
    residual = np.zeros_like(unique)
    np.add.at(residual, inverse, destination - source)
    return cKDTree(unique), residual / counts[:, None]


def apply(target, model):
    tree, residual = model
    unique, inverse = np.unique(target.reshape(-1, 3), axis=0, return_inverse=True)
    distances, indices = tree.query(unique, k=8, workers=1)
    weights = 1 / np.maximum(distances, 1)
    delta = (residual[indices] * weights[..., None]).sum(axis=1) / weights.sum(axis=1)[:, None]
    graded = np.rint(np.clip(unique + delta, 0, 255)).astype(np.uint8)
    full = graded[inverse].reshape(target.shape)
    return finish_spatial(target, full, STRENGTH), distances[inverse, -1]


def run(out):
    source = BASE / 'gros-film-b-cross-v2'
    report = checked_json(source / 'results.json', BASELINE_SHA)
    annotations = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    photos = {
        person: {
            light: {state: checked_rgb(source / f'{person}-{light}-{state}.png',
                                       report['inputs'][f'{person}-{light}']['hashes'][state])
                    for state in STATES}
            for light in LIGHTS
        }
        for person in GROUPS
    }
    models = {
        method: {
            donor: {light: {state: learn(states['off'], frame, method)
                            for state, frame in states.items()}
                    for light, states in lights.items()}
            for donor, lights in photos.items()
        }
        for method in METHODS
    }
    out.mkdir()
    records = {}
    for direction, ceiling in report['cases'].items():
        person, donor = direction.split('-from-')
        target = photos[person]['cold']['off']
        truth = checked_rgb(source / f'{direction}-expected.png', ceiling['expected_sha'])
        zones = masks(annotations[person]['cold'], annotations['screen'], target.shape[:2])
        details = {}
        for method in METHODS:
            prediction, hashes, support = {}, {}, {}
            for light in LIGHTS:
                states = {}
                for state in STATES:
                    image, distances = apply(target, models[method][donor][light][state])
                    if state == 'off':
                        np.testing.assert_array_equal(image, target)
                    states[state] = image
                    filename = f'{direction}-{method}-{light}-{state}.png'
                    hashes[filename] = save(out / filename, image)
                prediction[light] = states
                support[light] = {
                    name: float(np.median(distances.reshape(mask.shape)[mask]))
                    for name, mask in zones.items()
                }
            details[method] = {'scores': score(target, truth, prediction, zones),
                               'hashes': hashes, 'median_eighth_distance_rgb8': support}
        records[direction] = details
    json_to(out / 'results.json', {'source': BASELINE_SHA, 'region_ledger': REGIONS_SHA,
                                   'samples': SAMPLES, 'records': records})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
