"""Render a paired reference edit with explicitly declared intent, for development."""
import argparse
from itertools import permutations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from probe_beyond_rgb_matched_cast import compare
from probe_gros_garment_capture_cue import LOCAL, STATES
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_gros_screen_chroma import ARCHIVE, face_weight, local_edit, local_score
from probe_multilum_gray_probe import apply, chroma, valid
from probe_target_mirror_chroma import STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import finish_spatial


INTENT = {'cold': 'keep', 'warm': 'keep', 'matched': 'global', 'face_only': 'face'}
GARMENT_SHA = '549ecfb4d48573b384563038a7657465ddc8ae3e444e9a96051fc5f82676bc56'


def detect(detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5,
                                      minSize=(48, 48))
    return sorted(tuple(map(int, box)) for box in boxes)


def estimated_weight(image, box):
    x, y, w, h = box
    eye_y = y + .39 * h
    eyes = ((x + .30 * w, eye_y), (x + .70 * w, eye_y))
    return face_weight(image, eyes)


def replay(target, before, after, intent, source_faces, target_faces, fallback):
    if intent == 'keep' or np.array_equal(before, after):
        return target.copy(), {'status': 'keep'}
    if intent not in ('face', 'global'):
        raise ValueError('Unknown edit intent')
    if before.shape != after.shape:
        return fallback, {'status': 'mismatched_pair'}
    common = valid(before) & valid(after)
    if intent == 'face':
        if len(source_faces) != 1 or len(target_faces) != 1:
            return fallback, {'status': 'face_count'}
        common &= estimated_weight(before, source_faces[0]) >= .9
    samples = int(common.sum())
    if samples < 384:
        return fallback, {'status': 'insufficient_samples', 'samples': samples}
    delta = np.median(chroma(after[common]) - chroma(before[common]), axis=0)
    if intent == 'face':
        shifted = local_edit(target, delta, estimated_weight(target, target_faces[0]))
    else:
        shifted, _ = apply(target, delta)
    return finish_spatial(target, shifted, STRENGTH), {
        'status': intent, 'samples': samples, 'delta': delta.tolist()}


def run(gros, scope, garment, out):
    cv2.setNumThreads(1)
    source = checked_json(gros / 'results.json', ARCHIVE)
    local = checked_json(scope / 'results.json', LOCAL)
    prior = checked_json(garment / 'results.json', GARMENT_SHA)
    region_file = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_file) != source['regions_sha256'] or local['archive_sha256'] != ARCHIVE:
        raise ValueError('Changed source photos or region ledger')
    coords = json.loads(region_file.read_bytes())
    cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    # OpenCV on Windows cannot open this file from the project's Unicode path.
    with TemporaryDirectory(prefix='gros-face-') as directory:
        xml = Path(directory) / 'face.xml'
        xml.write_bytes(cascade_path.read_bytes())
        detector = cv2.CascadeClassifier(str(xml))
    if detector.empty():
        raise ValueError('Bundled face cascade could not be loaded')
    guides = {}
    for group in GROUPS:
        guides[group] = {state: checked_rgb(gros / 'guides' / f'{group}-{state}.png', expected)
                         for state, expected in source['admission'][group]['guide_sha256'].items()}
        guides[group]['face_only'] = checked_rgb(scope / 'local-guides' / f'{group}.png',
                                                  local['local_guides'][group])
    faces = {group: detect(detector, frames['cold']) for group, frames in guides.items()}
    example = guides[GROUPS[0]]['cold']
    blank = np.full_like(example, 128)
    double = np.concatenate((example, example), axis=1)
    edited = guides[GROUPS[0]]['face_only']
    controls = {}
    for name, target, expected in (('no_face', blank, 0), ('two_faces', double, 2)):
        detected = detect(detector, target)
        fallback = np.full_like(target, 127)
        result, trace = replay(target, example, edited, 'face', faces[GROUPS[0]], detected, fallback)
        controls[name] = dict(detected=detected, trace=trace,
                              passed=len(detected) == expected and trace['status'] == 'face_count'
                              and np.array_equal(result, fallback))
    identity, trace = replay(example, example, example, 'global', [], [], blank)
    controls['identity'] = dict(trace=trace, passed=np.array_equal(identity, example))
    out.mkdir()
    results = {}
    for target, donor in permutations(GROUPS, 2):
        name = f'{target}-from-{donor}'
        folder = out / name
        folder.mkdir()
        original = guides[target]['cold']
        old = source['results'][name]
        before = guides[donor]['cold']
        scoring = masks(coords[target]['cold'], coords['screen'], original.shape[:2])
        weight = face_weight(original, coords[target]['cold']['eyes'])
        base = {state: checked_rgb(gros / name / f'spatial13-{state}.png',
                                   old['layers']['spatial13']['hashes'][state])
                for state in STATES if state != 'face_only'}
        base['face_only'] = checked_rgb(scope / name / 'spatial13-face_only.png',
                                      local['cases'][name]['arms']['spatial13']['hashes']['face_only'])
        want_global = checked_rgb(gros / name / 'expected.png', old['expected_sha256'])
        want_local = checked_rgb(scope / name / 'expected-local.png', local['cases'][name]['expected_local_sha256'])
        frames, traces = {}, {}
        for state in STATES:
            frames[state], traces[state] = replay(original, before, guides[donor][state], INTENT[state],
                                                   faces[donor], faces[target], base[state])
        style = {key: compare(original, want_global, frames['cold'], frames['warm'], frames['matched'], region)
                 for key, region in scoring.items()}
        local_scorecard = local_score(original, want_local, frames, scoring, weight)
        identity = all(np.array_equal(frames[state], original) for state in ('warm', 'cold'))
        joint = (identity and traces['face_only']['status'] == 'face' and traces['matched']['status'] == 'global'
                 and all(row['both_pass'] for row in style.values()) and local_scorecard['passed'])
        results[name] = dict(
            traces=traces, style=style, local=local_scorecard, exact_preservation=identity, joint=joint,
            old_joint=local['cases'][name]['arms']['spatial13']['joint'],
            material_joint=prior['cases'][name]['arms']['garment']['joint'],
            hashes={state: save(folder / f'declared-{state}.png', result) for state, result in frames.items()},
            panel_sha=panel(folder / 'comparison.png', {'reference': guides[donor], '13': base, 'declared': frames,
                             'target': {'original': original, 'expected global': want_global,
                                        'expected local': want_local}}))
    passed_count = sum(row['joint'] for row in results.values())
    json_to(out / 'results.json', dict(
        script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name('gros-declared-edit-protocol.md')),
        sources=[ARCHIVE, LOCAL, GARMENT_SHA], cascade_sha=digest(cascade_path), opencv=cv2.__version__,
        strength=STRENGTH, faces=faces, intent=INTENT, controls=controls, cases=results, passed_count=passed_count,
        passed=passed_count >= 5 and all(row['passed'] for row in controls.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gros', 'scope', 'garment', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
