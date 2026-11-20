"""Hold donors fixed and render the held-out portrait with their development profile."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from probe_expanded_portrait_cross import measurement_masks
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_exact_color import RESPONSE_SHA, color_keys, medians, read_cases
from probe_rawtherapee_nearby_color import estimate
from probe_rawtherapee_portrait_cross import CLI_HASH, PROFILE_HASH, managed, score, section
from probe_rawtherapee_portrait_toning import RAW_HASH
from probe_rawtherapee_shared_response import REGION_SHA, STRENGTH, features, fit
from probe_sparks_shared_chroma import render as render_chroma
from probe_target_mirror_chroma import checked_json, json_to, save
from render_rawtherapee_sky_pair import check, disabled, render as render_raw
from spatial_render import finish_spatial

NEARBY_SHA = 'c3e4585f1eb259645c1dba60dfa5cf1d5f5af06f9d86933503e6c575dcb39a32'


def develop(raw, cli, parent, output):
    check(raw, RAW_HASH)
    check(cli, CLI_HASH)
    author = check(parent / 'profile-check.pp3', PROFILE_HASH)
    partial = b'\n'.join(section(author, name) for name in (b'Version', b'ColorToning', b'Resize'))
    images = {}
    for state, profile in (('off', disabled(partial, section=b'ColorToning')), ('on', partial)):
        path = output / state
        path.mkdir()
        pp3 = path / 'profile.pp3'
        pp3.write_bytes(profile)
        env = dict(os.environ, RT_SETTINGS=str(path / 'settings'), RT_CACHE=str(path / 'cache'), OMP_NUM_THREADS='1')
        render_raw(cli, raw, pp3, path / 'image.png', env)
        images[state] = managed(path / 'image.png')
    return images


def gradient(rgb):
    dy, dx = np.gradient(rgb @ LUMA)
    return np.hypot(dx, dy)


def register(old, new, boxes):
    if old.shape != new.shape:
        return {}, {name: {'accepted': False, 'reason': 'shape changed'} for name in boxes}
    old_edges, new_edges = gradient(old), gradient(new)
    registered, audit = {}, {}
    margin = 16
    for name, (x0, y0, x1, y1) in boxes.items():
        left, top, right, bottom = x0 - margin, y0 - margin, x1 + margin, y1 + margin
        template = old_edges[top:bottom, left:right].copy()
        template -= template.mean()
        window = new_edges[top - margin:bottom + margin, left - margin:right + margin]
        candidates = sliding_window_view(window, template.shape).copy()
        candidates -= candidates.mean(axis=(-2, -1), keepdims=True)
        denom = np.linalg.norm(candidates, axis=(-2, -1)) * np.linalg.norm(template)
        corr = np.divide(np.einsum('ijkl,kl->ij', candidates, template), denom,
                         out=np.full_like(denom, -1.), where=denom != 0)
        y, x = np.unravel_index(corr.argmax(), corr.shape)
        peak = float(corr[y, x])
        competitors = corr.copy()
        competitors[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = -1
        gap = peak - float(competitors.max())
        dx, dy = int(x - margin), int(y - margin)
        accepted = peak >= .65 and gap >= .05
        audit[name] = {'accepted': accepted, 'correlation': peak, 'gap': gap, 'offset': [dx, dy]}
        if accepted:
            registered[name] = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
    return registered, audit


def run(root, cli, response_dir, response_report, nearby_report, out):
    root, cli, out = (path.resolve() for path in (root, cli, out))
    response = checked_json(response_report, RESPONSE_SHA)
    nearby = checked_json(nearby_report, NEARBY_SHA)
    boxes = checked_json(Path(__file__).with_name('rawtherapee-portrait-toning-regions.json'), REGION_SHA)
    cases = {name: (before, expected) for name, before, expected, _ in read_cases(root, response_dir, response)}
    original, _ = cases.pop('cuckoo')
    # The donor expected files already include the 80% blend. Fit with no second blend.
    donors = [(before, expected) for before, expected in cases.values()]
    design = []
    for before, expected in donors:
        indices = np.linspace(0, before.shape[0] * before.shape[1] - 1, 8192, dtype=int)
        source = before.reshape(-1, 3)[indices]
        design.append((features(source), chroma(expected.reshape(-1, 3)[indices]) - chroma(source)))
    _, weights = fit(design)
    palette, values, _ = medians(np.concatenate([color_keys(before) for before, _ in donors]),
                                 np.concatenate([(chroma(expected) - chroma(before)).reshape(-1, 2)
                                                 for before, expected in donors]))
    out.mkdir()
    parent = root / 'rawtherapee-portrait-toning-inputs-v1'
    developed = develop(parent / 'DSC00178.ARW', cli, parent, out)
    before = developed['off']
    expected = finish_spatial(before, developed['on'], STRENGTH)
    registered, registration = register(original, before, boxes)
    masks = {label: mask for label, mask in measurement_masks(before, expected, registered).items() if mask.any()}
    colors, light = chroma(before), before @ LUMA
    change = (features(before.reshape(-1, 3)) @ weights).reshape(colors.shape)
    previous, _ = render_chroma(colors + change, light)
    unique, inverse = np.unique(color_keys(before), return_inverse=True)
    estimates, supported = estimate(palette, values, unique)
    rendered, _ = render_chroma(colors + estimates[inverse].reshape(colors.shape), light)
    supported = supported[inverse].reshape(light.shape)
    prediction = previous.copy()
    prediction[supported] = rendered[supported]

    draws = {'pixel-color': previous, 'nearby': prediction, 'expected': expected}
    runs = {name: score(before, expected, {'off': before, 'on': draws[name]}, masks)
            for name in ('pixel-color', 'nearby')}
    prior = nearby['cases']['cuckoo']['regions']
    for label, row in runs['nearby'].items():
        row['coverage'] = float(supported[masks[label]].mean())
        row['old_profile_mae'] = prior[label]['response_mae']
    new = runs['nearby']
    resolved = all(name in new and new[name]['passed'] and new[name]['coverage'] >= .8
                   and prior[name]['response_mae'] - new[name]['response_mae'] >= .25
                   for name in ('cheek', 'light_material'))
    json_to(out / 'results.json', {'old_nearby_report_sha256': NEARBY_SHA,
                                  'raw_sha256': RAW_HASH, 'cli_sha256': CLI_HASH, 'profile_sha256': PROFILE_HASH,
                                  'registered': registration, 'measurement_boxes': registered,
                                  'original_shape': list(original.shape), 'developed_shape': list(before.shape),
                                  'input_rgb_sha256': {k: sha256(v.tobytes()).hexdigest() for k, v in developed.items()},
                                  'output_sha256': {k: save(out / f'{k}.png', v) for k, v in draws.items()},
                                  'scores': runs, 'overgrading_resolved': resolved})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('root', 'cli', 'response_dir', 'response_report', 'nearby_report', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
