"""Ablate the reference appearance branch on the frozen matched-cast controls."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from probe_beyond_rgb_edit import make_masks, montage, verified
from probe_beyond_rgb_matched_cast import STATES, compare
from probe_beyond_rgb_paired import pinned_report
from probe_beyond_rgb_transfer import digest, save, write_json
from probe_expanded_background import WEIGHT, WEIGHT_SHA256
from probe_frozen_candidate_development_crosscheck import grade
from spatial_render import SpatialEngine
import json


BASELINE = '8ba8333f5bd89afd178523459a1dda02d8029d6fb1d4bd297e289c9baaf91d6c'
ARMS = {'stats': True, 'appearance': False}


@torch.inference_mode()
def run(prepared, previous, out):
    prior = pinned_report(previous, BASELINE)
    manifest = prepared / 'inputs.json'
    if digest(manifest) != prior['prepared_sha256'] or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed inputs or model checkpoint')
    records = json.loads(manifest.read_bytes())['records']
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != prior['checkpoint_sha256']:
        raise ValueError('Changed baseline spatial engine')
    initialize_replication_model(engine.model, engine.model.state_dict())
    state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite expanded weight')
    engine.model.load_state_dict(state, strict=True)
    engine.model.eval().requires_grad_(False)
    guides = {name: {kind: verified(previous / f'{name}-guide-{kind}.png', row['sha256'][kind])
                     for kind in STATES} for name, row in prior['references'].items()}
    out.mkdir()
    cases = {}
    for key, case in prior['cases'].items():
        target = verified(prepared / f'{key}.png', records[key]['image_sha256'])
        expected = verified(previous / f'{key}-expected.png', case['expected_sha256'])
        frozen = {kind: verified(previous / f'{key}-spatial13-{kind}.png',
                                 case['outputs_sha256'][f'{key}-spatial13-{kind}.png']) for kind in STATES}
        reference = guides[case['reference']]
        mask = make_masks(target, records[key]['patches'])
        rows = [[(label, image) for label, image in
                 [('Target', target), ('Reference base', reference['base']),
                  ('Reference light', reference['real']), ('Reference digital edit', reference['edit']),
                  ('Expected digital edit', expected)]],
                [(f'Research 13 / {kind}', image) for kind, image in frozen.items()]]
        comparisons = {}
        for label, disabled in ARMS.items():
            np.testing.assert_array_equal(grade(engine, target, target, target, disable_appearance=disabled)[0], target)
            artifacts, outputs = {}, {}
            for kind in STATES:
                result, change = grade(engine, target, reference[kind], frozen[kind], disable_appearance=disabled)
                name = f'{key}-{label}-{kind}.png'
                artifacts[name] = dict(sha256=save(out / name, result), **change)
                outputs[kind] = result
            rows.append([(f'{label} / {kind}', image) for kind, image in outputs.items()])
            comparisons[label] = dict(outputs=artifacts, identity_pass=True,
                scores={region: compare(target, expected, *outputs.values(), pixels)
                        for region, pixels in mask.items()})
        cases[key] = dict(reference=case['reference'], baseline=case['scores']['spatial13'], arms=comparisons,
                          montage_sha256=montage(out / f'{key}-comparison.png', rows))
    write_json(out / 'report.json', dict(baseline_sha256=BASELINE, weights_sha256=WEIGHT_SHA256,
                                        strength=prior['strength'], cases=cases))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for arg in ('prepared', 'previous', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
