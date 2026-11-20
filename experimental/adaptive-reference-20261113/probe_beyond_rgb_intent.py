"""Measure the error floor when the same two images have competing grading intents."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_paired import BASELINE_SHA, EDIT_SHA, pinned_report
from probe_beyond_rgb_transfer import LIGHTS, digest, save, write_json


LIMIT = 3.0


def compare(image, original, edited, mask):
    result = metrics(original, edited, original, image, mask)
    to_original = float(np.abs(image[mask].astype(np.int16) - original[mask]).mean())
    to_edited = result['on_matches_authored_mae']
    distance = result['original_matches_authored_mae']
    if to_original + to_edited + 1e-10 < distance:
        raise ValueError('Scoring violated the triangle bound')
    return dict(to_original_mae=to_original, to_edited_mae=to_edited, projection=result['projection'],
                both_error_limits=to_original <= LIMIT and to_edited <= LIMIT,
                preserves_and_transfers=to_original <= LIMIT and result['passed'])


def run(prepared, previous, edited, out):
    baseline = pinned_report(previous, BASELINE_SHA)
    prior = pinned_report(edited, EDIT_SHA)
    if digest(prepared / 'inputs.json') != baseline['prepared_sha256']:
        raise ValueError('Changed prepared inputs')
    out.mkdir()
    records = {}
    for name, event in prior['cases'].items():
        target_key, donor = name.split('-from-')
        spec = baseline['inputs'][target_key]
        target = verified(prepared / f'{target_key}.png', spec['image_sha256'])
        expected = verified(edited / f'{name}-expected.png', event['expected_sha256'])
        midpoint = np.rint((target.astype(np.int16) + expected) / 2).astype(np.uint8)
        midpoint_hash = save(out / f'{name}-midpoint.png', midpoint)
        masks = make_masks(target, spec['patches'])
        regions = {}
        for region, mask in masks.items():
            distance = float(np.abs(target[mask].astype(np.int16) - expected[mask]).mean())
            regions[region] = dict(intent_distance_mae=distance, worst_intent_lower_bound=distance / 2,
                                   incompatible_error_limits=distance > 2 * LIMIT,
                                   midpoint=compare(midpoint, target, expected, mask))
        guides, panels = {}, []
        for light in LIGHTS:
            ref_key = f'{donor}-{light}'
            ref_hash = prior['reference_sha256'][ref_key]
            reference = verified(edited / f'{ref_key}-edited.png', ref_hash)
            outputs = {}
            for pipeline in ('stable', 'spatial13'):
                filename = f'{target_key}-from-{ref_key}-{pipeline}.png'
                graded = verified(edited / filename, event['outputs_sha256'][filename])
                outputs[pipeline] = {region: compare(graded, target, expected, mask)
                                     for region, mask in masks.items()}
            guides[ref_key] = dict(reference_sha256=ref_hash, results=outputs)
            panels.append([('Same target / preserve', target), ('Same reference', reference),
                           ('Expected same edit', expected), ('Midpoint / offline only', midpoint)])
        records[name] = dict(target_sha256=spec['image_sha256'], expected_sha256=event['expected_sha256'],
                             midpoint_sha256=midpoint_hash, regions=regions, guides=guides,
                             panel_sha256=montage(out / f'{name}-comparison.png', panels))
    write_json(out / 'report.json', dict(clip_limit=LIMIT, baseline_sha256=BASELINE_SHA,
                                       edit_sha256=EDIT_SHA, cases=records))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('prepared', 'previous', 'edited', 'out'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
