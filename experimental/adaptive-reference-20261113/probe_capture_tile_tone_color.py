"""Evaluate a single-reference tone/chroma cue under whole-source holdout."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_beyond_rgb_matched_cast import compare
from probe_capture_chroma_edges import BEYOND_SHA, CAMERAS, TAGS, TARGETS, TRAIN_SHA
from probe_gros_screen_chroma import ARCHIVE
from probe_mid_matched_wall_cast import PARENT as MID_INPUTS, targets
from probe_multilum_gray_probe import chroma, valid
from probe_physical_blue_global_fit import linear
from probe_target_mirror_chroma import BASE, LUMA, checked_json, checked_rgb, digest, json_to, panel, save

MID_SHA = '9f1ca17f3b643017ec42f0a0003184ac3da455e5f93c40f93fe9c02669a0811a'
STATES = ('base', 'captured', 'edited')


def describe(image):
    scaled = np.array(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR))
    tiles = scaled.reshape(8, 32, 8, 32, 3).transpose(0, 2, 1, 3, 4).reshape(64, 1024, 3)
    rows = []
    for tile in tiles:
        rgb = tile[valid(tile)]
        if len(rgb) >= 820:
            y = linear(rgb) @ LUMA
            rows.append((float(y.std()), float(np.median(y)), np.median(chroma(rgb), axis=0)))
    if len(rows) < 16:
        return None
    flat = sorted(rows, key=lambda item: item[0])[:len(rows) // 2]
    quartiles = np.array_split(np.asarray([row[2] for row in sorted(flat, key=lambda item: item[1])]), 4)
    a, b, c, d = [np.median(chunk, axis=0) for chunk in quartiles]
    return np.r_[d - a, (b + c - a - d) / 2]


def acquire():
    camera = BASE / 'chromamill-camera-pair-first-v1'
    camera_rows = checked_json(camera / 'report.json', TRAIN_SHA)
    beyond = BASE / 'beyond-rgb-matched-cast-first-v1'
    beyond_rows = checked_json(beyond / 'report.json', BEYOND_SHA)
    gros = BASE / 'gros-portrait-light-transfer-v1'
    portraits = checked_json(gros / 'results.json', ARCHIVE)
    mid = BASE / 'mid-matched-wall-cast-v1'
    mid_rows = checked_json(mid / 'results.json', MID_SHA)
    groups = {'camera': {}, 'beyond': {}, 'gros': {}, 'mid': {}}
    for scene in TARGETS:
        for model in CAMERAS:
            record = camera_rows['scenes'][scene][model]['prepared_sha256']
            groups['camera'][f'{scene}-{model}'] = dict(zip(STATES, (
                checked_rgb(camera / f'{model}-{scene}-{tag}.png', record[tag]) for tag in TAGS), strict=True))
    for scene, record in beyond_rows['references'].items():
        groups['beyond'][scene] = {state: checked_rgb(beyond / f'{scene}-guide-{tag}.png', record['sha256'][tag])
                                    for state, tag in zip(STATES, ('base', 'real', 'edit'), strict=True)}
    for scene, record in portraits['admission'].items():
        groups['gros'][scene] = {state: checked_rgb(gros / f'guides/{scene}-{name}.png', record['guide_sha256'][name])
                                  for state, name in zip(STATES, ('cold', 'warm', 'matched'), strict=True)}
    groups['mid']['copyroom'] = {state: checked_rgb(mid / f'reference-{state}.png', sha)
                                 for state, sha in mid_rows['inputs'].items()}
    return groups, mid_rows


def classify(features, excluded):
    classes = {False: [], True: []}
    for family, records in features.items():
        if family == excluded:
            continue
        for values in records.values():
            for key, row in values.items():
                if row is not None:
                    classes[key == 'captured'].append(row)
    scale = np.maximum(np.std(classes[False] + classes[True], axis=0), 1e-6)
    centers = np.array([np.mean(classes[flag], axis=0) for flag in (False, True)])
    result = {}
    for scene, states in features[excluded].items():
        result[scene] = {}
        for label, value in states.items():
            distance = None if value is None else np.linalg.norm((value - centers) / scale, axis=1).tolist()
            result[scene][label] = dict(
                capture=None if distance is None else distance[1] < distance[0],
                vector=None if value is None else value.tolist(), distances=distance)
    return result


def render(mid, decisions, out):
    root = BASE / 'mid-matched-wall-cast-v1'
    ledger = checked_json(BASE / 'mid-gray-anchor-v1.json', MID_INPUTS)
    result = {}
    for name, (target, areas) in targets(ledger['scenes']).items():
        case = mid['cases'][name]
        authored = checked_rgb(root / f'{name}-expected.png', case['expected_sha'])
        original = {state: checked_rgb(root / f'{name}-13-{state}.png', sha)
                    for state, sha in case['models']['13']['hashes'].items()}
        output = {state: (target if decisions[state]['capture'] is True else frame) for state, frame in original.items()}
        scores = {key: compare(target, authored, *output.values(), area) for key, area in areas.items()}
        result[name] = dict(
            scores=scores, passed=all(row['both_pass'] and row['style']['off_matches_original_mae'] <= 3
                                      for row in scores.values()),
            hashes={state: save(out / f'{name}-{state}.png', frame) for state, frame in output.items()},
            comparison=panel(out / f'{name}-comparison.png', {'13': original, 'gated': output}))
    return result


def run(out):
    groups, mid = acquire()
    features = {family: {scene: {state: describe(image) for state, image in trio.items()}
                         for scene, trio in scenes.items()} for family, scenes in groups.items()}
    choices = {family: classify(features, family) for family in groups}
    summaries = {}
    for family, records in choices.items():
        summaries[family] = dict(real=sum(trio['captured']['capture'] is True for trio in records.values()),
                                 total=len(records), unavailable=sum(row['capture'] is None for trio in records.values()
                                                                    for row in trio.values()),
                                 false_activation=sum(row['capture'] is True for trio in records.values()
                                                      for label, row in trio.items() if label != 'captured'))
    out.mkdir()
    pictures = render(mid, choices['mid']['copyroom'], out)
    passed = all(row['real'] >= .75 * row['total'] and row['unavailable'] == 0 and row['false_activation'] == 0
                 for row in summaries.values()) and all(x['passed'] for x in pictures.values())
    json_to(out / 'results.json', dict(sources=[TRAIN_SHA, BEYOND_SHA, ARCHIVE, MID_SHA],
             choices=choices, summaries=summaries, pictures=pictures, passed=passed, script_sha=digest(Path(__file__)),
             protocol_sha=digest(Path(__file__).with_name('capture-tile-tone-color-protocol.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
