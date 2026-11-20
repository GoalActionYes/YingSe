"""Diagnose the remaining chroma loss with the saved profile-control renders."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_expanded_portrait_cross import measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_exact_color import RESPONSE_SHA, color_keys, read_cases
from probe_rawtherapee_nearby_color import estimate
from probe_rawtherapee_portrait_cross import managed, score
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, checked_rgb, json_to, save

PROFILE_REPORT_SHA = '10b3f9e4d50d1eae2036078c9f65805c2de04b6961b4b77ab23bb06c34206528'


def support_mask(root, response_dir, response_report, before):
    response = checked_json(response_report, RESPONSE_SHA)
    donors = [color_keys(original) for name, original, _, _ in read_cases(root, response_dir, response)
              if name != 'cuckoo']
    palette = np.unique(np.concatenate(donors))
    unique, inverse = np.unique(color_keys(before), return_inverse=True)
    _, supported = estimate(palette, np.zeros((len(palette), 2)), unique)
    return supported[inverse].reshape(before.shape[:2])


def describe(expected, actual, masks):
    all_energy = float(np.square(expected).sum())
    rows = {}
    for label, mask in masks.items():
        desired, seen = expected[mask], actual[mask]
        energy = float(np.square(desired).sum())
        paired = float((desired * seen).sum())
        rows[label] = {'pixels': int(mask.sum()), 'energy_share': energy / all_energy,
                       'projection': paired / energy if energy else None,
                       'projection_gap_contribution': (energy - paired) / all_energy,
                       'response_mae': float(np.abs(desired - seen).mean()) if mask.any() else None}
    return rows


def run(root, response_dir, response_report, profile_dir, profile_report, out):
    audit = checked_json(profile_report, PROFILE_REPORT_SHA)
    before = managed(profile_dir / 'off/image.png')
    if sha256(before.tobytes()).hexdigest() != audit['input_rgb_sha256']['off']:
        raise ValueError('Frozen working RGB changed')
    expected = checked_rgb(profile_dir / 'expected.png', audit['output_sha256']['expected'])
    previous = checked_rgb(profile_dir / 'nearby.png', audit['output_sha256']['nearby'])
    masks = measurement_masks(before, expected, audit['measurement_boxes'])
    masks = {label: mask for label, mask in masks.items() if mask.any()}
    supported = support_mask(root, response_dir, response_report, before)
    prior = audit['scores']['nearby']
    for label, mask in masks.items():
        np.testing.assert_allclose(supported[mask].mean(), prior[label]['coverage'], atol=1e-14, rtol=0)
    frozen = score(before, expected, {'off': before, 'on': previous}, masks)
    np.testing.assert_allclose(frozen['whole']['projection'], prior['whole']['projection'], atol=1e-14, rtol=0)

    truth = chroma(expected)
    original = chroma(before)
    ideal, limited = render(truth, before @ LUMA)
    unsupported_only = previous.copy()
    unsupported_only[~supported] = ideal[~supported]
    arms = {'unsupported-only': unsupported_only, 'all-pixels': ideal}
    files = {}
    scores = {'previous': frozen}
    out.mkdir()
    for label, picture in arms.items():
        files[label] = save(out / f'{label}.png', picture)
        scores[label] = score(before, expected, {'off': before, 'on': picture}, masks)
    strata = {'supported': supported, 'unsupported': ~supported}
    partition = describe(truth - original, chroma(previous) - original, strata)
    total_gap = sum(row['projection_gap_contribution'] for row in partition.values())
    np.testing.assert_allclose(total_gap, 1. - frozen['whole']['projection'], atol=1e-14, rtol=0)

    def qualified_pass(label):
        rows = scores[label]
        return rows['whole']['passed'] and all(row['passed'] for row in rows.values() if row['qualified'])

    json_to(out / 'results.json', {'profile_report_sha256': PROFILE_REPORT_SHA,
                                  'response_report_sha256': RESPONSE_SHA, 'partition': partition,
                                  'ideal_gamut_limited_fraction': limited, 'scores': scores,
                                  'unsupported_only_passed': qualified_pass('unsupported-only'),
                                  'all_pixels_passed': qualified_pass('all-pixels'), 'images': files})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'response_dir', 'response_report', 'profile_dir', 'profile_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
