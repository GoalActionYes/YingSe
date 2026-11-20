"""Swap face localization while holding paired-edit rendering and scoring fixed."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from probe_bg_real_legacy_skin import protector
from probe_gros_declared_edit import (
    ARCHIVE, GROUPS, LOCAL, checked_json, checked_rgb, digest, face_weight,
    json_to, local_score, masks, panel, replay, save,
)
from probe_reference_skin_category_gate import predictions
from probe_target_mirror_chroma import ROOT


PARENT_SHA = 'b6ea7a97e1e94058b0f7c4afd4526931c89a8014105dc9b32e7c5d2c352f711d'


def proposals(model, image):
    category, _, face = predictions(model, image)
    area = np.uint8((category == 3) & (face >= .5))
    _, _, stats, _ = cv2.connectedComponentsWithStats(area, connectivity=8)
    scale_x, scale_y = image.shape[1] / area.shape[1], image.shape[0] / area.shape[0]
    return [(round(x * scale_x), round(y * scale_y), round(w * scale_x), round(h * scale_y))
            for x, y, w, h, size in stats[1:] if size >= area.size * .001]


def run(gros, scope, previous, out):
    cv2.setNumThreads(1)
    source = checked_json(gros / 'results.json', ARCHIVE)
    local = checked_json(scope / 'results.json', LOCAL)
    prior = checked_json(previous / 'results.json', PARENT_SHA)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if source['regions_sha256'] != digest(region_file):
        raise ValueError('Changed scoring regions')
    coords = json.loads(region_file.read_bytes())
    originals = {group: checked_rgb(gros / 'guides' / f'{group}-cold.png',
                                   source['admission'][group]['guide_sha256']['cold']) for group in GROUPS}
    edited = {group: checked_rgb(scope / 'local-guides' / f'{group}.png', local['local_guides'][group])
              for group in GROUPS}
    model = protector(ROOT / 'experimental/skin-protection-20261019/reference_color_app/skin.py')
    faces = {group: proposals(model, image) for group, image in originals.items()}
    sample = originals[GROUPS[0]]
    blank = np.full_like(sample, 128)
    double = np.concatenate((sample, sample), axis=1)
    controls = {}
    for name, target, count in (('no_face', blank, 0), ('two_faces', double, 2)):
        detected = proposals(model, target)
        fallback = np.full_like(target, 127)
        actual, trace = replay(target, sample, edited[GROUPS[0]], 'face', faces[GROUPS[0]], detected, fallback)
        controls[name] = dict(faces=detected, passed=len(detected) == count and trace['status'] == 'face_count'
                              and np.array_equal(actual, fallback))
    out.mkdir()
    cases = {}
    for target, donor in permutations(GROUPS, 2):
        name = f'{target}-from-{donor}'
        folder = out / name
        folder.mkdir()
        original = originals[target]
        archived = prior['cases'][name]
        frozen = {state: checked_rgb(previous / name / f'declared-{state}.png', sha)
                  for state, sha in archived['hashes'].items()}
        fallback = checked_rgb(scope / name / 'spatial13-face_only.png',
                               local['cases'][name]['arms']['spatial13']['hashes']['face_only'])
        truth = checked_rgb(scope / name / 'expected-local.png', local['cases'][name]['expected_local_sha256'])
        actual, trace = replay(original, originals[donor], edited[donor], 'face', faces[donor],
                               faces[target], fallback)
        target_coords = coords[target]['cold']
        weight = face_weight(original, target_coords['eyes'])
        scoring = masks(target_coords, coords['screen'], original.shape[:2])
        old_score = local_score(original, truth, frozen, scoring, weight)
        if old_score != archived['local']:
            raise ValueError('Previous local score changed')
        report = local_score(original, truth, dict(cold=frozen['cold'], face_only=actual), scoring, weight)
        preserved = all(np.array_equal(frozen[state], original) for state in ('warm', 'cold'))
        joint = (preserved and archived['traces']['matched']['status'] == 'global'
                 and all(item['both_pass'] for item in archived['style'].values())
                 and trace['status'] == 'face' and report['passed'])
        cases[name] = dict(old=old_score, old_joint=archived['joint'], trace=trace, local=report, joint=joint,
                           output_sha=save(folder / 'face.png', actual),
                           panel_sha=panel(folder / 'comparison.png', {
                               'reference': {'original': originals[donor], 'local edit': edited[donor]},
                               'target': {'original': original, '13': fallback, 'expected': truth},
                               'replay': {'old locator': frozen['face_only'], 'face segment': actual},
                           }))
    count = sum(case['joint'] for case in cases.values())
    kept = all(case['joint'] for case in cases.values() if case['old_joint'])
    json_to(out / 'results.json', dict(
        parent_sha=PARENT_SHA, model_sha=model.MODEL_SHA256, opencv=cv2.__version__,
        script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
            'gros-face-segment-location-protocol.md')), faces=faces, controls=controls, cases=cases,
        count=count, retained_old_passes=kept,
        passed=bool(count >= 5 and kept and all(control['passed'] for control in controls.values()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('gros', 'scope', 'previous', 'out', 'locator_dependencies'):
        parser.add_argument(argument, type=Path)
    args = vars(parser.parse_args())
    sys.path.append(str(args.pop('locator_dependencies').resolve(strict=True)))
    run(**args)
