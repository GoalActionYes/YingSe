"""Test whether a second visible skin region identifies physical lighting."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from probe_multilum_gray_probe import chroma, valid
from probe_portrait_chroma_tone_cue import (
    AUTHOR_BOXES, GROS_SOURCE, GROUPS, LOCAL_SOURCE, PARENT, PARENT_SHA, STATES,
    checked_json, checked_rgb, classify, digest, fit, json_to, masks, read,
)


NECK = {
    '01': {'cold': (453, 364, 464, 385), 'warm': (454, 368, 465, 389)},
    '02': {'cold': (367, 431, 407, 446), 'warm': (367, 432, 407, 447)},
    '03': {'cold': (356, 404, 410, 436), 'warm': (356, 416, 410, 448)},
}
AUTHOR_NECK = (762, 555, 789, 568)


def crop(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def sample(region, minimum):
    pixels = region[valid(region)]
    return dict(pixels=len(pixels), total=region.size // 3,
                median=np.median(chroma(pixels), axis=0).tolist() if len(pixels) >= minimum else None)


def measure(face, neck, neck_qualified):
    face_stat = sample(face, 384)
    neck_stat = sample(neck, 128)
    median = face_stat['median']
    difference = None
    if neck_qualified and median is not None and neck_stat['median'] is not None:
        difference = (np.asarray(median) - neck_stat['median']).tolist()
    return dict(face=face_stat, neck=neck_stat, neck_qualified=neck_qualified,
                arms={'median': dict(values={'median': median} if median is not None else None),
                      'difference': dict(values={'difference': difference} if difference is not None else None)})


def author_data():
    previous = checked_json(PARENT / 'results.json', PARENT_SHA)
    off, _ = read(PARENT / 'off.tif', False)
    on, mask = read(PARENT / 'on.tif', True)
    if ([sha256(rgb.tobytes()).hexdigest() for rgb in (off, on)] != previous['rgb_sha256']
            or sha256(mask.tobytes()).hexdigest() != previous['mask_pixels_sha256']):
        raise ValueError('Author RGB or mask changed')
    face = AUTHOR_BOXES['child_face'][0]
    neck_mask = crop(mask, AUTHOR_NECK)
    neck_same = np.array_equal(crop(off, AUTHOR_NECK), crop(on, AUTHOR_NECK))
    neck_qualified = not neck_mask.any() and neck_same
    return dict(face_high_weight=int((crop(mask, face) >= 204).sum()),
                neck_nonzero=int(np.count_nonzero(neck_mask)), neck_peak=int(neck_mask.max()),
                neck_same=neck_same,
                states={state: measure(crop(rgb, face), crop(rgb, AUTHOR_NECK), neck_qualified)
                        for state, rgb in (('off', off), ('on', on))})


def decide(people, author, arm):
    examples = {group: {state: item['arms'][arm] for state, item in states.items()}
                for group, states in people.items()}
    held_out = {}
    for group, states in examples.items():
        model = fit(examples, group, arm)
        held_out[group] = {state: classify(record, arm, model) for state, record in states.items()}
    warm = sum(row['warm']['physical'] is True for row in held_out.values())
    failures = {f'{group}:{state}': 'unavailable' if row['physical'] is None else 'false_positive'
                for group, states in held_out.items() for state, row in states.items()
                if state != 'warm' and row['physical'] is not False}
    entire_set = fit(examples, None, arm)
    author_decisions = {state: classify(item['arms'][arm], arm, entire_set)
                        for state, item in author['states'].items()}
    passed = warm == len(GROUPS) and not failures and author['face_high_weight'] >= 128
    return dict(held_out=held_out, correct_warm=warm, negative_failures=failures,
                author=author_decisions, passed=passed and author_decisions['on']['physical'] is False)


def run(gros, local, out):
    original = checked_json(gros / 'results.json', GROS_SOURCE)
    extra = checked_json(local / 'results.json', LOCAL_SOURCE)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_file) != original['regions_sha256'] or extra['archive_sha256'] != GROS_SOURCE:
        raise ValueError('Changed reference region or photo archive')
    regions = json.loads(region_file.read_bytes())
    people = {}
    for group in GROUPS:
        guides = {state: checked_rgb(gros / 'guides' / f'{group}-{state}.png',
                                     original['admission'][group]['guide_sha256'][state])
                  for state in STATES if state != 'face_only'}
        guides['face_only'] = checked_rgb(local / 'local-guides' / f'{group}.png', extra['local_guides'][group])
        people[group] = {}
        for state, picture in guides.items():
            placement = 'warm' if state == 'warm' else 'cold'
            face = masks(regions[group][placement], regions['screen'], picture.shape[:2])
            neck = crop(picture, NECK[group][placement])
            qualified = state != 'face_only' or np.array_equal(neck, crop(guides['cold'], NECK[group]['cold']))
            people[group][state] = measure(picture[face['cheek_left'] | face['cheek_right']], neck, qualified)
    author = author_data()
    decisions = {arm: decide(people, author, arm) for arm in ('median', 'difference')}
    out.mkdir()
    json_to(out / 'results.json', dict(
        script_sha256=digest(Path(__file__)), protocol_sha256=digest(Path(__file__).with_name(
            'portrait-secondary-skin-cue-protocol.md')),
        gros_sha256=GROS_SOURCE, local_sha256=LOCAL_SOURCE, author_sha256=PARENT_SHA,
        regions_sha256=digest(region_file), neck_boxes=NECK, author_neck_box=AUTHOR_NECK,
        cases=people, authored=author, decisions=decisions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gros', 'local', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
