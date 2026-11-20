"""Render existing workbenches against matched physical and edited wall casts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_beyond_rgb_matched_cast import compare
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_gros_screen_chroma import ARCHIVE
from probe_multilum_gray_probe import apply, chroma, valid
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save,
)
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze


PARENT = '766c8e5df51edfc4c11d4d1b1e4cce2b20ea2fbadb226057b69ba7eeecf9408d'


def box(image, limits):
    x0, y0, x1, y1 = limits
    result = np.zeros(image.shape[:2], bool)
    result[y0:y1, x0:x1] = True
    return result


def material(pair, area):
    before, after = pair
    core = valid(before) & valid(after) & area
    count = int(core.sum())
    return dict(coverage=count / int(area.sum()), pixels=count,
                delta=np.median(chroma(after[core]) - chroma(before[core]), axis=0).tolist() if count else None)


def guides(base, real, bounds):
    left = box(base, bounds['wall_left'])
    change = material((base, real), left)
    delta = np.asarray(change['delta'])
    edited, clipped = apply(base, delta)
    core = left & valid(base) & valid(real) & valid(edited)
    if not core.any():
        raise ValueError('No comparable wall pixels')
    first, second = [np.median(chroma(item[core]) - chroma(base[core]), axis=0) for item in (real, edited)]
    distance = float(np.linalg.norm(first - second))
    coverage = int(core.sum()) / int(left.sum())
    qualified = bool(coverage >= .80 and np.linalg.norm(first) >= .08 and distance <= .025)
    materials = {key: {label: material((base, image), box(base, bounds[key]))
                       for label, image in (('captured', real), ('edited', edited))}
                 for key in bounds}
    return dict(base=base, captured=real, edited=edited), delta, dict(
        materials=materials, delta=delta.tolist(), distance=distance, coverage=coverage,
        clipped_fraction=clipped, qualified=qualified)


def targets(ledger):
    living = ledger['everett_living2']
    home = checked_rgb(BASE / 'mid-new-room-ranges-v1/everett_living2-14.jpg', living['members']['14']['sha256'])
    home_zones = {name: box(home, limits) for name, limits in living['boxes'].items()}
    gros = BASE / 'gros-portrait-light-transfer-v1'
    manifest = checked_json(gros / 'results.json', ARCHIVE)
    region_ledger = Path(__file__).with_name('gros-portrait-light-regions.json')
    if digest(region_ledger) != manifest['regions_sha256']:
        raise ValueError('Changed human rating regions')
    person = GROUPS[0]
    portrait = checked_rgb(gros / f'guides/{person}-cold.png',
                           manifest['admission'][person]['guide_sha256']['cold'])
    coords = json.loads(region_ledger.read_bytes())
    home_zones['whole'] = np.ones(home.shape[:2], bool)
    return {'living': (home, home_zones), 'portrait': (portrait, masks(coords[person]['cold'], coords['screen'], portrait.shape[:2]))}


def run(out):
    torch.set_num_threads(4)
    ledger_path = BASE / 'mid-gray-anchor-v1.json'
    ledger = checked_json(ledger_path, PARENT)['scenes']
    room = ledger['14n_copyroom1']
    if room['selected']['directions'] != [0, 16] or not room['accepted']:
        raise ValueError('Changed wall control eligibility')
    source = BASE / 'mid-gray-anchor-new-jpg-v1'
    images = [checked_rgb(source / f'14n_copyroom1-{number}.jpg', room['members'][str(number)]['sha256'])
              for number in (0, 16)]
    frames, delta, admission = guides(*images, room['boxes'])
    if not admission['qualified']:
        raise ValueError(f'Digital cast did not match actual wall: {admission}')
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed spatial checkpoint')
    out.mkdir()
    paths = {name: out / f'reference-{name}.png' for name in frames}
    inputs = {name: save(paths[name], rgb) for name, rgb in frames.items()}
    results = {}
    for name, (target, regions) in targets(ledger).items():
        target_path = out / f'{name}-target.png'
        target_sha = save(target_path, target)
        expected, clipping = apply(target, delta)
        expected = finish_spatial(target, expected, STRENGTH)
        expected_sha = save(out / f'{name}-expected.png', expected)
        layers = {'stable': {}, '13': {}}
        for label, guide in frames.items():
            layers['stable'][label] = apply_mapping(target, analyze(target_path, paths[label]), STRENGTH)
            layers['13'][label] = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
        rows = {}
        for model, outputs in layers.items():
            scores = {label: compare(target, expected, *outputs.values(), area) for label, area in regions.items()}
            rows[model] = dict(scores=scores, passed=all(score['both_pass'] and score['style']['off_matches_original_mae'] <= 3
                                                       for score in scores.values()),
                               hashes={label: save(out / f'{name}-{model}-{label}.png', rgb)
                                       for label, rgb in outputs.items()})
        results[name] = dict(target_sha=target_sha, expected_sha=expected_sha, clipping=clipping, models=rows,
                             board=panel(out / f'{name}-comparison.png', {
                                 'reference': frames, 'target': {'input': target, 'expected digital edit': expected}, **layers}))
    json_to(out / 'results.json', dict(parent=PARENT, strength=STRENGTH, stable=algorithm_fingerprint(),
            spatial=CHECKPOINT, reference=admission, inputs=inputs, cases=results,
            script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
                'mid-matched-wall-cast-protocol.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
