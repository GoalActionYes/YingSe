"""Test the value of knowing the actual pre-edit reference, with mismatch controls."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_transfer import LIGHTS, SCENES, STRENGTH, digest, save, write_json
from reference_color_app.aligned_reference import fit_aligned
from spatial_render import finish_spatial


BASELINE_SHA = 'ad2b10f42558ec951c430ae823438444e783b568d1c05ec16e13c099d03ffa1e'
EDIT_SHA = 'adb53a3fb2cc0eeb465b0c444aa684b5d521638cf87ce2f864b057743521872e'


def apply_fit(image, fit):
    if not fit['accepted']:
        return image.copy()
    coefficients = np.asarray(fit['matrix'])
    working = (image.astype(np.float64) / 255) @ coefficients[:3] + coefficients[3]
    graded = np.rint(np.clip(working, 0, 1) * 255).astype(np.uint8)
    return finish_spatial(image, graded, STRENGTH)


def pinned_report(folder, expected):
    path = folder / 'report.json'
    if digest(path) != expected:
        raise ValueError(f'Changed evidence in {folder.name}')
    return json.loads(path.read_bytes())


def run(prepared, previous, edited, out):
    baseline = pinned_report(previous, BASELINE_SHA)
    edits = pinned_report(edited, EDIT_SHA)
    if edits['baseline_sha256'] != BASELINE_SHA or digest(prepared / 'inputs.json') != baseline['prepared_sha256']:
        raise ValueError('Changed sealed input manifest')
    out.mkdir()
    inputs = baseline['inputs']
    plain, styled = {}, {}
    for key in inputs:
        plain[key] = verified(prepared / f'{key}.png', inputs[key]['image_sha256'])
        styled[key] = verified(edited / f'{key}-edited.png', edits['reference_sha256'][key])

    fits = {}
    for scene in SCENES:
        for light, other_light in (LIGHTS, LIGHTS[::-1]):
            guide = f'{scene}-{light}'
            alternate = f'{scene}-{other_light}'
            fits[guide] = {
                'true_pair': fit_aligned(plain[guide], styled[guide]),
                'same_image': fit_aligned(plain[guide], plain[guide]),
                'wrong_capture': fit_aligned(plain[alternate], styled[guide]),
            }

    cases = {}
    for case, prior in edits['cases'].items():
        key, donor = case.split('-from-')
        target = plain[key]
        expected = verified(edited / f'{case}-expected.png', prior['expected_sha256'])
        masks = make_masks(target, inputs[key]['patches'])
        guides, panels, outputs = {}, [], []
        for light in LIGHTS:
            guide = f'{donor}-{light}'
            images, scores = {}, {}
            for kind, fit in fits[guide].items():
                result = apply_fit(target, fit)
                images[kind] = result
                path = out / f'{case}-{light}-{kind}.png'
                scores[kind] = {
                    'sha256': save(path, result),
                    'changes_target_mae': float(np.abs(result.astype(np.int16) - target).mean()),
                    'regions': {label: metrics(target, expected, target, result, mask)
                                for label, mask in masks.items()},
                }
            outputs.append(images['true_pair'])
            guides[guide] = scores
            panels.append([('Target', target), ('Expected', expected), ('True reference pair', images['true_pair']),
                           ('Same-image control', images['same_image']), ('Wrong capture', images['wrong_capture'])])
        cases[case] = dict(guides=guides, panel_sha256=montage(out / f'{case}-comparison.png', panels),
                           switch_light_mae=float(np.abs(outputs[1].astype(np.int16) - outputs[0]).mean()))
    write_json(out / 'report.json', dict(baseline_sha256=BASELINE_SHA, edit_sha256=EDIT_SHA, strength=STRENGTH,
                                       fits=fits, cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for arg in ('prepared', 'previous', 'edited', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
