"""Measure content-held-out edit detection using local luma–chroma relationships."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_expanded_portrait_cross import BASELINE_SHA, load_inputs
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_rawtherapee_portrait_cross import CLI_HASH, PROFILE_HASH, managed, section
from probe_target_mirror_chroma import checked_json, json_to
from render_rawtherapee_sky_pair import SOURCES, check, disabled, render

STATES = ('off', 'on')
QUANTILES = (.1, .5, .9)


def endpoints(root, parent, cli, out):
    base = checked_json(parent / 'results.json', BASELINE_SHA)
    images = load_inputs(root, parent, base['results'])
    check(cli, CLI_HASH)
    author = check(root / 'rawtherapee-portrait-toning-inputs-v1/profile-check.pp3', PROFILE_HASH)
    active = b'\n'.join(section(author, name) for name in (b'Version', b'ColorToning', b'Resize'))
    profiles = {'off': disabled(active, section=b'ColorToning'), 'on': active}
    for name, (extension, raw_hash, _) in SOURCES.items():
        raw = root / 'rawtherapee-sky-pair-v1' / f'{name}.{extension}'
        check(raw, raw_hash)
        images[name] = {}
        for state, profile in profiles.items():
            folder = out / (name + '-' + state)
            folder.mkdir()
            settings = folder / 'profile.pp3'
            settings.write_bytes(profile)
            image = folder / 'image.png'
            env = dict(os.environ, RT_SETTINGS=str(folder / 'settings'), RT_CACHE=str(folder / 'cache'),
                       OMP_NUM_THREADS='1')
            render(cli, raw, settings, image, env)
            images[name][state] = managed(image)
    return images


def describe(rgb):
    small = np.array(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))
    brightness = small @ LUMA
    color = chroma(small)
    absolute = np.quantile(np.dstack((brightness, color)), QUANTILES, axis=(0, 1)).ravel()
    local = []
    for top in range(0, 256, 64):
        for left in range(0, 256, 64):
            box = np.s_[top:top + 64, left:left + 64]
            y = brightness[box].ravel()
            pixels = color[box].reshape(-1, 2)
            order = np.argsort(y, kind='stable')
            local.append(pixels[order[-1024:]].mean(0) - pixels[order[:1024]].mean(0))
    return {'whole': absolute, 'local': np.quantile(local, QUANTILES, axis=0).ravel()}


def classify(features, arm):
    rows = {}
    for group, states in features.items():
        training = np.stack([items[state][arm] for name, items in features.items() if name != group
                             for state in STATES])
        average = training.mean(0)
        scale = np.maximum(training.std(0), 1.)
        design = (training - average) / scale
        labels = np.tile([-1., 1.], len(features) - 1)
        weights = np.linalg.solve(design.T @ design + np.eye(design.shape[1]), design.T @ labels)
        rows[group] = {}
        for state, values in states.items():
            decision = float(((values[arm] - average) / scale) @ weights)
            rows[group][state] = {'decision': decision, 'correct': (decision > 0) == (state == 'on')}
    return {'cases': rows, 'correct': {state: sum(row[state]['correct'] for row in rows.values()) for state in STATES}}


def run(root, parent, cli, out):
    root, parent, cli, out = (p.resolve() for p in (root, parent, cli, out))
    out.mkdir()
    images = endpoints(root, parent, cli, out)
    features = {name: {state: describe(rgb) for state, rgb in states.items()} for name, states in images.items()}
    results = {arm: classify(features, arm) for arm in ('whole', 'local')}
    local, whole = (results[arm]['correct'] for arm in ('local', 'whole'))
    each_type = all(any(results['local']['cases'][name][state]['correct'] for name in names)
                    for names in (('cuckoo', 'mixed'), tuple(SOURCES)) for state in STATES)
    passed = each_type and min(local.values()) >= 3 and sum(local.values()) >= sum(whole.values()) + 2
    audit = {name: {'rgb_sha256': {state: sha256(rgb.tobytes()).hexdigest() for state, rgb in states.items()},
                    'editor_chroma_mae': float(np.abs(chroma(states['on']) - chroma(states['off'])).mean())}
             for name, states in images.items()}
    json_to(out / 'results.json', {'parent_sha256': BASELINE_SHA, 'results': results, 'inputs': audit, 'passed': passed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'parent', 'cli', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
