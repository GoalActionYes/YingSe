"""Test whether tone-conditioned skin chroma separates lighting from edits."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_gros_screen_chroma import ARCHIVE as GROS_SOURCE
from probe_multilum_gray_probe import LUMA, chroma, valid
from probe_physical_blue_global_fit import linear
from probe_playraw_mixed_light_transfer import PARENT, PARENT_SHA, read
from probe_target_mirror_chroma import checked_json, checked_rgb, digest, json_to


LOCAL_SOURCE = 'c057e8ec8789531a5df8dfe0689994b28704ed6a64ee1008efb170106e3b5547'
STATES = ('cold', 'warm', 'matched', 'face_only')
ARMS = ('median', 'tonal_difference')
AUTHOR_BOXES = {'face': [(170, 213, 291, 379)],
                'near_hand': [(144, 463, 285, 608)], 'far_hand': [(350, 480, 450, 567)],
                'child_face': [(742, 484, 810, 549)]}


def measure(pixels):
    usable = pixels[valid(pixels)]
    coverage = len(usable) / len(pixels)
    if len(usable) < 384:
        return dict(pixels=len(usable), coverage=coverage, values=None)
    light = linear(usable) @ LUMA
    ordered = np.array_split(light.argsort(kind='stable'), 3)
    color = chroma(usable)
    levels = np.array([np.median(light[index]) for index in ordered])
    medians = np.array([np.median(color[index], axis=0) for index in ordered])
    qualified = levels[2] - levels[0] >= .01
    return dict(pixels=len(usable), coverage=coverage, band_light=levels.tolist(), band_chroma=medians.tolist(),
                values={'median': np.median(color, axis=0).tolist(),
                        'tonal_difference': (medians[2] - medians[0]).tolist()} if qualified else None)


def fit(records, excluded, arm):
    examples = {True: [], False: []}
    for group, states in records.items():
        if group == excluded:
            continue
        for state, item in states.items():
            if item['values'] is not None:
                examples[state == 'warm'].append(item['values'][arm])
    if min(map(len, examples.values())) == 0:
        return None
    scale = np.maximum(np.std(examples[True] + examples[False], axis=0), 1e-6)
    return scale, [np.mean(examples[key], axis=0) for key in (True, False)]


def classify(example, arm, model):
    if example['values'] is None or model is None:
        return dict(physical=None)
    scale, centers = model
    rgb = np.asarray(example['values'][arm])
    physical, other = [float(np.linalg.norm((rgb - center) / scale)) for center in centers]
    return dict(physical=physical < other, distance_physical=physical, distance_other=other)


def as_author():
    previous = checked_json(PARENT / 'results.json', PARENT_SHA)
    off, _ = read(PARENT / 'off.tif', False)
    on, mask = read(PARENT / 'on.tif', True)
    if ([sha256(rgb.tobytes()).hexdigest() for rgb in (off, on)] != previous['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != previous['mask_pixels_sha256']):
        raise ValueError('Author RGB or mask changed')
    result = {}
    for name, [(x0, y0, x1, y1)] in AUTHOR_BOXES.items():
        area = np.s_[y0:y1, x0:x1]
        core = mask[area] >= 204
        item = {state: measure(image[area].reshape(-1, 3)) for state, image in (('off', off), ('on', on))}
        item['high_weight_pixels'] = int(core.sum())
        item['core_coverage'] = {state: float(valid(image[area])[core].mean()) if core.any() else None
                                 for state, image in (('off', off), ('on', on))}
        result[name] = item
    return result


def run(gros, local, out):
    original = checked_json(gros / 'results.json', GROS_SOURCE)
    extra = checked_json(local / 'results.json', LOCAL_SOURCE)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_file) != original['regions_sha256'] or extra['archive_sha256'] != GROS_SOURCE:
        raise ValueError('Changed reference region or photo archive')
    boxes = json.loads(region_file.read_bytes())
    people = {}
    for group in GROUPS:
        records = {}
        for state in STATES:
            if state == 'face_only':
                picture = checked_rgb(local / 'local-guides' / f'{group}.png', extra['local_guides'][group])
            else:
                expected = original['admission'][group]['guide_sha256'][state]
                picture = checked_rgb(gros / 'guides' / f'{group}-{state}.png', expected)
            regions = masks(boxes[group]['warm' if state == 'warm' else 'cold'], boxes['screen'], picture.shape[:2])
            records[state] = measure(picture[regions['cheek_left'] | regions['cheek_right']])
        people[group] = records
    authored = as_author()
    decisions = {}
    for arm in ARMS:
        result = {}
        for group, states in people.items():
            model = fit(people, group, arm)
            result[group] = {state: classify(item, arm, model) for state, item in states.items()}
        correct = sum(row['warm']['physical'] is True for row in result.values())
        mistaken = [f'{group}:{state}' for group, states in result.items() for state, row in states.items()
                    if state != 'warm' and row['physical'] is not False]
        model = fit(people, None, arm)
        edit = authored['child_face']
        author = {state: classify(edit[state], arm, model) for state in ('off', 'on')}
        passes = (correct == 3 and not mistaken and edit['high_weight_pixels'] >= 128
                  and author['on']['physical'] is False)
        decisions[arm] = dict(held_out=result, correct_warm=correct, negative_failures=mistaken,
                              author=author, passed=passes)
    out.mkdir()
    json_to(out / 'results.json', dict(script_sha256=digest(Path(__file__)), gros_sha256=GROS_SOURCE,
                                      local_sha256=LOCAL_SOURCE, author_sha256=PARENT_SHA,
                                      regions_sha256=digest(region_file), cases=people, authored=authored,
                                      decisions=decisions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gros', 'local', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
