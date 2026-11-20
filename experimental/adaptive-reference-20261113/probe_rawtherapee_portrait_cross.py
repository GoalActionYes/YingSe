"""Cross-reference two portraits rendered with the same editor color-toning switch."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from probe_playraw_mixed_light_transfer import CHECKPOINT, STRENGTH, chroma, stats
from probe_rawtherapee_portrait_toning import CLI_HASH, PROFILE_HASH
from probe_target_mirror_chroma import digest, json_to, save
from render_rawtherapee_sky_pair import check, disabled, render
from spatial_render import SpatialEngine, finish_spatial
from triplet_color import to_srgb

ORF_SHA = '7b4cc05a958acd8e10a161d1b2faf6cae9d25a37de28550b0680b34c14c38217'
CUCKOO = {'off': '1983eaa0f5a33761cc00c86c98e0b65abb17dbe1749c639a5ac4f3b8b06b5b58',
          'on': '4376790260cc5b41fd83576742f757f7d1d18e4473d38fa9a6b9b52b3199f0c6'}


def section(profile, name):
    expression = rb'(?ms)^\[' + name + rb'\]\r?\n.*?(?=^\[|\Z)'
    contents, = re.findall(expression, profile)
    return contents


def managed(path):
    with Image.open(path) as image:
        if image.mode != 'RGB' or not image.info.get('icc_profile'):
            raise ValueError('Expected ICC-tagged RGB')
        return np.array(to_srgb(image)[0])


def render_second(root, cli, out):
    raw = root / 'playraw-mixed-light-skin-v1/P5020458.ORF'
    check(raw, ORF_SHA)
    author = check(root / 'rawtherapee-portrait-toning-inputs-v1/profile-check.pp3', PROFILE_HASH)
    profile = b'\n'.join(section(author, name) for name in (b'Version', b'ColorToning', b'Resize'))
    off = disabled(profile, section=b'ColorToning')
    files = {}
    for state, data in (('off', off), ('on', profile)):
        directory = out / ('mixed-author-' + state)
        directory.mkdir()
        pp3 = directory / 'profile.pp3'
        pp3.write_bytes(data)
        image = directory / 'image.png'
        env = dict(os.environ, RT_SETTINGS=str(directory / 'settings'), RT_CACHE=str(directory / 'cache'),
                   OMP_NUM_THREADS='1')
        render(cli, raw, pp3, image, env)
        files[state] = image
    return files


def score(target, expected, outputs, masks):
    observed = chroma(outputs['on']) - chroma(outputs['off'])
    authored = chroma(expected) - chroma(target)
    baseline = np.abs(outputs['off'].astype(np.int16) - target)
    readings = {}
    for name, mask in masks.items():
        wanted = float(np.abs(authored[mask]).mean())
        qualified = int(mask.sum()) >= 128 and wanted >= 1
        measured = stats(observed, authored, mask)
        drift = float(baseline[mask].mean())
        passed = (qualified and .75 <= measured['projection'] <= 1.25
                  and measured['response_mae'] <= 3 and drift <= 3)
        readings[name] = {'edited_chroma_mae': wanted, 'off_rgb_mae': drift, 'qualified': qualified,
                          'passed': passed, **measured}
    return readings


@torch.inference_mode()
def run(root, cli, out):
    root, cli, out = (path.resolve() for path in (root, cli, out))
    check(cli, CLI_HASH)
    out.mkdir()
    cuckoo = {state: root / 'rawtherapee-portrait-toning-v1' / state / 'image.png' for state in CUCKOO}
    for state, path in cuckoo.items():
        check(path, CUCKOO[state])
    pairs = {'cuckoo': cuckoo, 'mixed': render_second(root, cli, out)}
    images = {name: {state: managed(path) for state, path in pair.items()} for name, pair in pairs.items()}
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed model')
    predictions = {}
    for name, alternative in [('cuckoo', 'mixed'), ('mixed', 'cuckoo')]:
        target = images[name]['off']
        other = images[alternative]
        outputs = {state: finish_spatial(target, engine.prepare(target, ref).apply(target), STRENGTH)
                   for state, ref in other.items()}
        predictions[name] = outputs, {state: save(out / f'{name}-from-{alternative}-{state}.png', value)
                                     for state, value in outputs.items()}

    region_file = Path(__file__).with_name('rawtherapee-portrait-toning-regions.json')
    registered = json.loads(region_file.read_bytes())
    results = {}
    for name, (outputs, files) in predictions.items():
        target, edited = (images[name][state] for state in ('off', 'on'))
        expected = finish_spatial(target, edited, STRENGTH)
        guide = np.zeros(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(image, guide)['neutral_proxy'] for image in (target, expected)))
        masks = {'whole': ~guide}
        if neutral.any():
            masks['neutral'] = neutral
        if name == 'cuckoo':
            for region, (x0, y0, x1, y1) in registered.items():
                mask = guide.copy()
                mask[y0:y1, x0:x1] = True
                masks[region] = mask
        scores = score(target, expected, outputs, masks)
        expected_hash = save(out / f'{name}-expected.png', expected)
        results[name] = {'scores': scores, 'predictions': files, 'expected_sha256': expected_hash,
                         'inputs_rgb_sha256': {state: sha256(rgb.tobytes()).hexdigest()
                                               for state, rgb in images[name].items()},
                         'passed': all(scores.get(region, {}).get('passed', False) for region in ('whole', 'neutral'))}
    json_to(out / 'results.json', {'checkpoint_sha256': CHECKPOINT, 'cli_sha256': CLI_HASH,
                                  'operator_profile_sha256': PROFILE_HASH, 'mixed_raw_sha256': ORF_SHA,
                                  'region_sha256': digest(region_file), 'results': results,
                                  'passed': all(row['passed'] for row in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cli', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
