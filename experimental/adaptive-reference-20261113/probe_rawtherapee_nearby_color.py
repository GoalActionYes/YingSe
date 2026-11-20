"""Render a bounded nearby-color estimate from the other editor pairs."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np

from probe_rawtherapee_exact_color import RESPONSE_SHA, REGION_SHA, color_keys, medians, read_cases
from probe_expanded_portrait_cross import measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_portrait_cross import score
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import checked_json, json_to, save

RADIUS = 6
NEIGHBORS = 8


def rgb(keys):
    return ((keys[:, None] >> np.array([16, 8, 0])) & 255).astype(np.int16)


def estimate(donor_keys, donor_values, target_keys):
    lookup = np.full(1 << 24, -1, dtype=np.int32)
    lookup[donor_keys] = np.arange(len(donor_keys))
    target = rgb(target_keys)
    counts = np.zeros(len(target), dtype=np.int16)
    total = np.zeros((len(target), 2))
    weight_sum = np.zeros(len(target))
    offsets = sorted((sum(c * c for c in delta), delta)
                     for delta in product(range(-RADIUS, RADIUS + 1), repeat=3))
    pending = np.arange(len(target))
    for squared, delta in offsets:
        if squared > RADIUS**2 or not pending.size:
            break
        shifted = target[pending] + np.asarray(delta, dtype=np.int16)
        valid = np.all((shifted >= 0) & (shifted <= 255), axis=1)
        selected = pending[valid]
        indices = lookup[color_keys(shifted[valid])]
        match = indices >= 0
        selected, indices = selected[match], indices[match]
        weight = 1 / max(float(np.sqrt(squared)), 1.)
        total[selected] += weight * donor_values[indices]
        weight_sum[selected] += weight
        counts[selected] += 1
        pending = pending[counts[pending] < NEIGHBORS]
    np.divide(total, weight_sum[:, None], out=total, where=weight_sum[:, None] > 0)
    return total, counts > 0


def run(root, response_dir, response_report, out):
    baseline = checked_json(response_report, RESPONSE_SHA)
    registered = checked_json(Path(__file__).with_name('rawtherapee-portrait-toning-regions.json'), REGION_SHA)
    cases = list(read_cases(root, response_dir, baseline))
    keys = [color_keys(before) for _, before, _, _ in cases]
    changes = [(chroma(expected) - chroma(before)).reshape(-1, 2) for _, before, expected, _ in cases]
    out.mkdir()
    report = {}
    regressions = []
    for held, (name, before, expected, previous) in enumerate(cases):
        donors = [k for k in range(len(cases)) if k != held]
        palette, values, _ = medians(np.concatenate([keys[k] for k in donors]),
                                     np.concatenate([changes[k] for k in donors]))
        unique, inverse = np.unique(keys[held], return_inverse=True)
        estimates, supported = estimate(palette, values, unique)
        supported = supported[inverse].reshape(before.shape[:2])
        correction = estimates[inverse].reshape(*before.shape[:2], 2)
        rendered, _ = render(chroma(before) + correction, before @ LUMA)
        picture = previous.copy()
        picture[supported] = rendered[supported]
        masks = measurement_masks(before, expected, registered if name == 'cuckoo' else {})
        masks = {label: mask for label, mask in masks.items() if mask.any()}
        rows = score(before, expected, {'off': before, 'on': picture}, masks)
        old_error = np.abs(chroma(previous) - chroma(expected)).mean(-1)
        new_error = np.abs(chroma(picture) - chroma(expected)).mean(-1)
        for label, row in rows.items():
            mask = masks[label]
            measured = mask & supported
            old = float(old_error[mask].mean())
            row.update(coverage=float(measured.sum() / mask.sum()), previous_mae=old,
                       previous_supported_mae=float(old_error[measured].mean()) if measured.any() else None,
                       new_supported_mae=float(new_error[measured].mean()) if measured.any() else None)
            if row['qualified'] and row['response_mae'] > old + .25:
                regressions.append([name, label])
        report[name] = {'regions': rows, 'prediction_sha256': save(out / f'{name}-nearby.png', picture)}
    grass = report['cuckoo']['regions']['green_background']
    grass_pass = grass['coverage'] >= .8 and grass['previous_mae'] - grass['response_mae'] >= 1.
    portraits_pass = all(report[name]['regions']['whole']['passed'] and
                         all(row['passed'] for row in report[name]['regions'].values() if row['qualified'])
                         for name in ('cuckoo', 'mixed'))
    sky_pass = any(report[name]['regions']['whole']['passed'] for name in ('P1005231', 'R6_06283'))
    json_to(out / 'results.json', {'baseline_sha256': RESPONSE_SHA, 'cases': report, 'regressions': regressions,
                                  'grass_pass': grass_pass, 'passed': grass_pass and portraits_pass
                                  and sky_pass and not regressions})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'response_dir', 'response_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
