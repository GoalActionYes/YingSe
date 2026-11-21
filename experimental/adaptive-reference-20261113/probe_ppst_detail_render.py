"""Apply PPST color residuals while retaining the target's original high frequencies."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_ppst_first_portrait import gradient_correlation
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, json_to, save


PARENT_SHA = 'a477cc59bde27b184b37e39d29b048a0556716bc4b3ff8fb4760403f4169bad5'
EPSILON = (.02 * 255) ** 2
RADIUS = 30
STATES = ('off', 'on')
METHODS = ('guided_image', 'guided_residual')


def guided(guide, values):
    return cv2.ximgproc.guidedFilter(guide, values, RADIUS, EPSILON)


def transfer(target, candidate, detail):
    target = target.astype(np.float32)
    candidate = candidate.astype(np.float32)
    if detail:
        output = target + guided(target, candidate - target)
    else:
        output = guided(target, candidate)
    clipped = float(((output < 0) | (output > 255)).mean())
    return np.rint(np.clip(output, 0, 255)).astype(np.uint8), clipped


def run(out):
    cv2.setNumThreads(1)
    photos = BASE / 'gros-film-b-cross-v2'
    manifest = checked_json(photos / 'results.json', BASELINE_SHA)
    roi = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    prior = BASE / 'ppst-first-portrait-v2'
    baseline = checked_json(prior / 'results.json', PARENT_SHA)
    case = manifest['cases']['01-from-02']
    original = checked_rgb(photos / '01-cold-off.png', manifest['inputs']['01-cold']['hashes']['off'])
    boxes = masks(roi['01']['cold'], roi['screen'], original.shape[:2])
    inputs = {light: {
        state: checked_rgb(prior / f'{light}-{state}.png', baseline['media'][f'{light}-{state}']['sha256'])
        for state in STATES} for light in LIGHTS}
    self_image = checked_rgb(prior / 'same.png', baseline['self_control']['sha256'])
    np.testing.assert_array_equal(transfer(original, original, True)[0], original)
    out.mkdir()

    outputs, summaries = {}, {}
    for mode in METHODS:
        detail = mode == 'guided_residual'
        predictions, media = {}, {}
        for light in LIGHTS:
            predictions[light] = {}
            for state in STATES:
                name = f'{light}-{state}'
                frame, clip = transfer(original, inputs[light][state], detail)
                predictions[light][state] = frame
                media[name] = {'image_sha256': save(out / f'{mode}-{name}.png', frame),
                               'clip_fraction': clip, 'structure': gradient_correlation(original, frame)}
        same, clip = transfer(original, self_image, detail)
        summaries[mode] = {'media': media, 'same': {
            'image_sha256': save(out / f'{mode}-same.png', same), 'clip_fraction': clip,
            'structure': gradient_correlation(original, same),
            'mae': float(np.abs(same.astype(np.int16) - original).mean()),
        }}
        outputs[mode] = predictions

    expected = checked_rgb(photos / '01-from-02-expected.png', case['expected_sha'])
    if score(original, expected, inputs, boxes) != baseline['scores']:
        raise ValueError('Frozen PPST score changed')
    for mode, predictions in outputs.items():
        summaries[mode]['scores'] = score(original, expected, predictions, boxes)
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'radius': RADIUS,
                                   'epsilon': EPSILON, 'methods': summaries})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
